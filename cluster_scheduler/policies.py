"""Custom sb3-contrib policy with set-attention machine embeddings.

Why this exists: the default ``MlpPolicy`` treats per-machine observation
slots as ordered positional features, so it cannot learn a shared
"score this machine" function and its action space is hard-coded to the
training ``num_machines``. This module provides a permutation-invariant,
cross-cluster-generalizable alternative.

Components:

- :class:`SetAttentionExtractor`: reshapes the flat
  :class:`cluster_scheduler.featurizers.SetFeaturizer` observation to a
  set of ``MAX_MACHINES`` rows, embeds each with a shared MLP, injects
  the current job's context, and runs multi-head self-attention with
  ``is_active`` as the key-padding mask. Outputs flattened per-machine
  embeddings plus the ``is_active`` bits so the policy heads can do
  masked pooling.
- :class:`SetMaskablePolicy`: subclasses sb3-contrib's
  :class:`MaskableActorCriticPolicy`, replaces the default
  ``action_net`` / ``value_net`` with set-aware heads, and keeps the
  shared ``mlp_extractor`` as an identity passthrough. The actor head
  applies a single shared linear layer to each machine's embedding to
  produce per-machine logits (Pointer-network-style). The critic head
  masked-mean-pools the active machines' embeddings and projects to a
  scalar value.

Trained with :func:`cluster_scheduler.train.train_maskable_ppo` by
passing ``policy_kwargs={"policy_class": SetMaskablePolicy}`` — or via
the ``--policy set_attention`` CLI flag that wires everything up.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor

from cluster_scheduler.featurizers import SetFeaturizer


class SetAttentionExtractor(BaseFeaturesExtractor):
    """Self-attention over machines with shared per-machine embedding.

    Obs layout (matches :class:`SetFeaturizer`):

    - ``obs[:, :MAX*D_M]``: flattened per-machine features.
    - ``obs[:, MAX*D_M : MAX*D_M + D_J]``: current-job features.
    - ``obs[:, MAX*D_M + D_J : MAX*D_M + D_J + D_G]``: global features.

    Output (flat): ``[B, MAX*embed_dim + MAX]`` — per-machine embeddings
    concatenated with their ``is_active`` bits. Policy heads split this
    back into the two pieces.

    Args:
        observation_space: The env's flat Box observation space.
        max_machines: Must equal :attr:`SetFeaturizer.MAX_MACHINES`.
        d_m: Per-machine feature dim (defaults to ``SetFeaturizer.D_M``).
        d_j: Current-job feature dim.
        d_g: Global feature dim.
        embed_dim: Size of each machine's learned embedding.
        n_heads: Attention heads per layer.
        n_layers: Stacked attention layers.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        max_machines: int = SetFeaturizer.MAX_MACHINES,
        d_m: int = SetFeaturizer.D_M,
        d_j: int = SetFeaturizer.D_J,
        d_g: int = SetFeaturizer.D_G,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        features_dim = max_machines * embed_dim + max_machines
        super().__init__(observation_space, features_dim=features_dim)

        self.max_machines = int(max_machines)
        self.d_m = int(d_m)
        self.d_j = int(d_j)
        self.d_g = int(d_g)
        self.embed_dim = int(embed_dim)

        self.machine_embed = nn.Sequential(
            nn.Linear(d_m, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.job_embed = nn.Linear(d_j + d_g, embed_dim)
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim, n_heads, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mxd = self.max_machines * self.d_m
        batch = obs.shape[0]

        machines = obs[:, :mxd].reshape(batch, self.max_machines, self.d_m)
        job = obs[:, mxd : mxd + self.d_j]
        glob = obs[:, mxd + self.d_j : mxd + self.d_j + self.d_g]
        is_active = machines[:, :, 4]  # [B, MAX]; matches SetFeaturizer's layout

        m_embed = self.machine_embed(machines)  # [B, MAX, embed_dim]
        job_ctx = self.job_embed(torch.cat([job, glob], dim=-1))  # [B, embed_dim]
        m_embed = m_embed + job_ctx.unsqueeze(1)  # broadcast context across machines

        # True => ignore (padded) in MultiheadAttention.
        key_padding_mask = is_active < 0.5

        # If an episode ever presented no active machines we'd produce NaNs
        # (attention over an entirely masked set is undefined). The env never
        # lets this happen — at least one machine is always active — but
        # a belt-and-braces zero-out on fully-masked rows keeps training safe.
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)

        for attn, norm in zip(self.attn_layers, self.norms):
            attn_out, _ = attn(
                m_embed, m_embed, m_embed, key_padding_mask=key_padding_mask
            )
            attn_out = torch.where(
                all_masked.unsqueeze(-1), torch.zeros_like(attn_out), attn_out
            )
            m_embed = norm(m_embed + attn_out)

        flat = m_embed.reshape(batch, self.max_machines * self.embed_dim)
        return torch.cat([flat, is_active], dim=-1)


class _SetActorHead(nn.Module):
    """Shared linear score applied per machine to produce MAX action logits."""

    def __init__(self, max_machines: int, embed_dim: int):
        super().__init__()
        self.max_machines = int(max_machines)
        self.embed_dim = int(embed_dim)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        total = self.max_machines * self.embed_dim
        embed = latent_pi[:, :total].reshape(-1, self.max_machines, self.embed_dim)
        return self.score(embed).squeeze(-1)  # [B, MAX]


class _SetValueHead(nn.Module):
    """Masked mean-pool of per-machine embeddings → MLP → scalar V(s)."""

    def __init__(self, max_machines: int, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.max_machines = int(max_machines)
        self.embed_dim = int(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent_vf: torch.Tensor) -> torch.Tensor:
        total = self.max_machines * self.embed_dim
        embed = latent_vf[:, :total].reshape(-1, self.max_machines, self.embed_dim)
        is_active = latent_vf[:, total : total + self.max_machines]
        mask = is_active.unsqueeze(-1)  # [B, MAX, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)  # avoid div by zero
        pooled = (embed * mask).sum(dim=1) / denom  # [B, embed_dim]
        return self.mlp(pooled)  # [B, 1]


class SetMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy with set-attention extractor and per-machine scoring.

    The default ``MlpPolicy`` pipeline is:
    ``obs → features_extractor → mlp_extractor → action_net / value_net``.
    We keep the same chain but replace three pieces:

    1. ``features_extractor`` → :class:`SetAttentionExtractor` (set-aware
       embeddings + active-mask bits).
    2. ``mlp_extractor`` → identity passthrough (empty ``net_arch``).
    3. ``action_net`` / ``value_net`` → :class:`_SetActorHead` /
       :class:`_SetValueHead`, which know how to slice the attention output
       back into ``[B, MAX, embed_dim]`` and the ``is_active`` bits.

    SB3's ``MaskableCategoricalDistribution`` consumes the actor logits
    exactly as before, so masking + PPO update proceed unchanged.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Any,
        *args: Any,
        attention_kwargs: Optional[dict[str, Any]] = None,
        value_hidden_dim: int = 128,
        **kwargs: Any,
    ):
        fex_kwargs = dict(attention_kwargs or {})
        kwargs["features_extractor_class"] = SetAttentionExtractor
        kwargs["features_extractor_kwargs"] = fex_kwargs
        # The default MLP ``net_arch`` is ignored by this policy — heads are
        # set-aware and operate directly on the extractor's output. Silently
        # drop if the user forwarded one (CLI does by default).
        kwargs.pop("net_arch", None)
        self._value_hidden_dim = int(value_hidden_dim)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # Identity passthrough: latent_pi == latent_vf == features.
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=dict(pi=[], vf=[]),
            activation_fn=nn.Identity,
            device=self.device,
        )

    def _build(self, lr_schedule: Any) -> None:
        # Let SB3 build default action_net (Linear(features_dim → n_actions))
        # and value_net, then swap them out for the set-aware heads. This
        # ensures everything the parent class expects is initialised first
        # (distribution, optimizer, init weights), then we replace the two
        # final projections and re-init the optimizer.
        super()._build(lr_schedule)

        ext = self.features_extractor
        max_m = ext.max_machines
        embed = ext.embed_dim

        self.action_net = _SetActorHead(max_m, embed).to(self.device)
        self.value_net = _SetValueHead(
            max_m, embed, hidden_dim=self._value_hidden_dim
        ).to(self.device)

        # Rebuild the optimizer so it picks up the new head parameters and
        # drops the old action_net/value_net params SB3 constructed.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )
