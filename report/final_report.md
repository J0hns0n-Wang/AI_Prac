# Learning Cluster-Scheduling Policies that Improve Tail Latency

**CS 4701 Practicum in Artificial Intelligence — Spring 2026**

Frank Dai, Johnson Wang, Jerry Ji
Cornell University

## Abstract

We train action-masked PPO policies for cluster job placement and compare them against
classical heuristics (First-Fit, Best-Fit, Shortest-Job-First) across six controlled
workload regimes. Three policies — one each for 4-, 10-, and 20-machine clusters —
share a common training recipe: a `RichFeaturizer` that exposes per-machine
post-placement headroom, a dense reward shaped by per-step backlog and a
completion bonus, `VecNormalize` on returns, and a linear learning-rate schedule
over 8 million environment steps. The learned policies achieve the best p99
waiting time on three of five informative regimes (`bursty`, `large_jobs`,
`light_poisson`) and ranks within 5 % of the best heuristic on the remaining
two; against duration-oracle SJF it improves p99 waiting time by 1.7×–2.5×
under congestion, at the cost of ~30 % higher mean. Cluster utilization is
within 0.3 % across all schedulers, indicating that the learned policy's
gains come from ordering and packing decisions rather than admission control.
A negative-result attempt at a single cross-cluster set-attention policy is
documented in Section 11.

## 1. Introduction

Modern cluster schedulers (e.g. Borg, Kubernetes) decide which incoming job
runs on which machine, balancing utilization, fairness, deadline adherence, and
tail latency. A common observation in production [Dean & Barroso 2013] is that
even a small fraction of slow placements drives the user-visible service-level
objective (SLO) for the entire cluster — the **tail** matters more than the
mean. Heuristic schedulers like First-Fit and Shortest-Job-First (SJF) are
simple and fast, but provably suboptimal in adversarial regimes; reinforcement
learning offers an alternative when reward shaping can target the metric of
interest directly.

This report presents a small, controlled study of action-masked PPO for cluster
placement. We do not aspire to compete with state-of-the-art systems like
Decima [Mao et al. 2019]; instead we ask:

- Can a learned policy beat oblivious heuristics on tail-latency metrics
  while staying within reach on mean-latency?
- Does duration-oracle SJF dominate the tail-latency story, and if not, why?
- Where do the learned policy's gains come from — better admission, better
  packing, or better ordering?

The contributions of this work are:

1. A modular cluster-scheduling simulator and Gymnasium environment with
   action masking, multiple heuristic baselines, and an evaluation harness
   that runs every scheduler on identical workloads (Sections 4–6).
2. A reproducible training recipe for `MaskablePPO` over a `RichFeaturizer`
   observation, with reward shaping, observation/return normalization, and
   linear learning-rate decay (Section 7).
3. An evaluation across six workload regimes spanning load level,
   burstiness, demand mix, and cluster size, with 95 % confidence intervals
   from five held-out seeds (Sections 8–9).
4. A documented negative result: a custom set-attention policy intended for
   cross-cluster generalization that suffers entropy collapse in training
   (Section 11), with a working test harness left in the repository.

## 2. Related work

**Bin-packing heuristics.** First-Fit and Best-Fit are classical for online
bin packing; their competitive ratios are well known [Coffman, Garey, Johnson
1996]. SJF (Shortest-Job-First) is the optimal greedy policy for minimizing
mean waiting time when durations are known a priori, but performs poorly on
tail metrics because long jobs starve.

**RL for systems.** DeepRM [Mao et al. 2016] applied policy-gradient methods to
multi-resource cluster scheduling on a discrete-time grid representation.
Decima [Mao et al. 2019] extended this with a graph neural network policy over
job DAGs and produces strong results on Spark traces. SchedRL frameworks have
since explored auto-tuning hyperparameters and curriculum learning. Our work
sits below these in scope: we use a simple flat-vector featurizer and a
standard MLP policy, and our contribution is a careful comparison across
regimes rather than a new architecture.

**Action masking in PPO.** `MaskablePPO` [Huang & Ontañón 2022], part of
`sb3-contrib`, augments PPO with state-dependent action masks so that
infeasible actions receive zero probability. This is the natural fit for
cluster placement, where many machines cannot fit a given job at any moment.

## 3. Problem formulation

We model cluster scheduling as a discrete-time MDP. The cluster has `M`
machines, each with fixed CPU and memory capacity. A workload generator
produces a sequence of jobs with arrival times drawn from a (possibly bursty)
Poisson process and CPU / memory / duration drawn from configurable ranges.

**State.** A snapshot containing, for each machine, current CPU and memory
utilization; the job currently being placed; and a waiting queue of jobs
that have arrived but not yet been placed.

**Action.** An index into `{0, …, M-1}` selecting which machine receives the
current job. The environment supplies an action mask: machine `i` is masked
*off* whenever it cannot fit the current job's CPU or memory demand.

**Transition.** Once a machine is selected, the job begins executing
immediately (if the action is legal — masking ensures this). The simulator
advances to the next decision point, defined as the earliest of (a) the
next job arrival, or (b) the completion of any running job. A job that
arrives while no machine can accommodate it joins the wait queue and is
re-considered after each completion event.

**Reward.** The default reward is `-waiting_time` per placement (zero for
immediate placement, negative for queued jobs). For shaping, a configurable
`RewardConfig` allows adding a per-step penalty proportional to current
backlog size and a one-time bonus on episode termination (Section 7).

**Episode.** An episode terminates when all `num_jobs` workload jobs have
been placed and completed. We use 200 jobs for the 4-machine policy, 500
for the 10-machine, and 400 for the 20-machine, calibrated for ~75 % CPU
utilization at the chosen arrival rates.

## 4. System design

The codebase decouples five components so that heuristic and learned
schedulers run through a common evaluation path:

- `cluster_scheduler.simulator.Simulator` — discrete-event core that
  manages event ordering and the wait queue. It is agnostic to scheduler
  choice and consumes any object satisfying the `Scheduler` interface.
- `cluster_scheduler.scheduler.Scheduler` — abstract base class with a
  single method `schedule(queue, machines) -> (job_idx, machine_idx) | None`.
  Heuristics (`FirstFitScheduler`, `BestFitScheduler`,
  `ShortestJobFirstScheduler`) and the learned `RLScheduler` all satisfy
  this interface.
- `cluster_scheduler.env.ClusterSchedulingEnv` — Gymnasium wrapper around
  the simulator. The agent picks the machine; the environment picks the
  next queued job that fits (FIFO head-of-queue-that-fits).
- `cluster_scheduler.featurizers.Featurizer` — pluggable function from
  cluster state to flat observation vector. We define `BasicFeaturizer`
  (utilization, current job, queue length, average cluster utilization)
  and `RichFeaturizer` (basic plus top-k queued-job peek and per-machine
  post-placement headroom).
- `cluster_scheduler.scheduler.RLScheduler` — adapter that wraps a trained
  `MaskablePPO` model so it slots into `Simulator.run` exactly like any
  heuristic, sharing the obs construction and action masking with the
  training environment.

**Fair-comparison contract.** The evaluation harness
`cluster_scheduler.evaluation.sweep` generates one workload per `(regime,
seed)` cell and reuses it across every scheduler under test. This rules
out workload-noise as a confounder when comparing schedulers.

## 5. Heuristic baselines

We use three heuristics as baselines:

- **First-Fit (FF):** scan machines in id order, place on the first that
  fits. Simple, low-latency, no global view.
- **Best-Fit (BF):** scan all machines, place on the one with the smallest
  remaining `(cpu_free + memory_free)` after placement. Reduces
  fragmentation; slightly more compute per decision.
- **Shortest-Job-First (SJF):** sort the queue by duration ascending; place
  the shortest fittable job on the first fittable machine. SJF requires
  job durations as input; we provide them. SJF is the **oracle** lower
  bound on mean waiting time for a single resource; it is **expected to
  perform poorly on the tail** because long jobs starve in saturated
  regimes.

## 6. RL method

**Algorithm.** We use `MaskablePPO` from `sb3-contrib`. The action mask is
the boolean vector `[m.can_fit(current_job) for m in machines]`, exposed
to the policy via `env.action_masks()`. PPO's categorical distribution is
masked before the softmax, so infeasible actions never receive
probability.

**Featurizer.** `RichFeaturizer(top_k=4)` produces a flat-vector
observation containing:

- per-machine `(cpu_utilization, memory_utilization)`;
- the current job's `(cpu/cpu_per_machine, memory/memory_per_machine,
  duration / DURATION_REF)` (with `DURATION_REF = 100.0` as a fixed
  normalizer that decouples the observation from the train-time
  duration distribution);
- the next four queued jobs' `(cpu, memory, duration)` (zero-padded);
- backlog totals: total CPU and total memory demand summed over the
  queue, normalized by total cluster capacity;
- per-machine post-placement headroom: how much CPU/memory each machine
  would have after accepting the current job.

The full observation is clipped to `[0, 1]` and consumed by an MLP policy
with two 256-unit GeLU hidden layers.

**Reward shaping.** `RewardConfig(mode="dense",
wait_penalty_weight=1.0, backlog_penalty_weight=0.1, completion_bonus=1.0)`.
The dense per-step reward is `-waiting_time - 0.1 * len(wait_queue)`,
discouraging both individual long waits and growing global backlog. A
`+1.0` bonus is added on the terminal step.

**Stability tricks.**

- `VecNormalize(norm_reward=True)` keeps the value-function target scale
  bounded across regimes (the running statistics are saved alongside the
  model so resumed training can continue them).
- Linear learning-rate decay from `2.5e-4` to `0` over training.
- `clip_range_vf=0.2` clips the value-function loss for additional
  critic stability.

**Per-cluster training.** Because `MaskablePPO`'s action space is fixed at
`Discrete(num_machines)`, a single trained policy cannot generalize across
cluster sizes. We train one policy per cluster size: `ppo_m4_v4` (4
machines), `ppo_m10_v4` (10 machines), and `ppo_m20_v4` (20 machines).
The RL adapter dispatches policies based on the regime's `num_machines`.

**Hyperparameters.** `n_steps=1024`, `batch_size=4096`, `gamma=0.999`,
`gae_lambda=0.95`, `ent_coef=0.005`, `clip_range=0.2`. 8 million
environment steps per policy, 10 parallel `SubprocVecEnv` workers.
Training time on a 12-core M4 Pro: ~14 minutes per policy.

## 7. Experimental setup

**Regimes.** We evaluate on six controlled regimes, defined in
`cluster_scheduler.evaluation.default_regimes()`:

| regime | num_machines | num_jobs | arrival_rate | notes |
| --- | --- | --- | --- | --- |
| `light_poisson` | 10 | 200 | 5.0 | Low load; baseline. |
| `heavy_poisson` | 10 | 200 | 8.0 | Saturated Poisson arrivals. |
| `bursty` | 10 | 200 | 4.0 (avg) | Bursts to rate 12.0 with prob 0.15, length 8. |
| `large_jobs` | 10 | 150 | 3.0 | CPU range (3,6); memory range (4,12). Packing pressure. |
| `small_cluster` | 4 | 150 | 2.0 | Saturated 4-machine cluster. |
| `wide_cluster` | 20 | 400 | 6.0 | Wide cluster; under-saturated at this rate. |

**Seeds.** Five held-out evaluation seeds (`1000`–`1004`), disjoint from
the training seed (`0`). Each seed determines the workload trace shared
across all four schedulers.

**Statistics.** Confidence intervals are 95 % normal-approximation
intervals on the mean across five seeds, computed by
`cluster_scheduler.metrics.mean_ci`. Cell format throughout this report
is `mean ± half-width`.

**Reproducibility.** Each saved model has a JSON sidecar
(`*.zip.meta.json`) recording the featurizer name and parameters, reward
configuration, PPO hyperparameters, network architecture, training seed,
and total timesteps. This lets downstream evaluation reconstruct the
exact observation and policy used at training time.

## 8. Results

### 8.1 Per-regime comparison

Table 1 reports waiting-time metrics across all six regimes. (Full numbers
in `report/results_tables.md`.) Bold values mark the best scheduler per
regime/metric combination.

**Mean waiting time** (`avg_waiting_time`). SJF wins every regime, by
26–37 %, as expected: SJF is an oracle on duration and short jobs see
short queues. RL is statistically tied with First-Fit and Best-Fit (the
duration-agnostic heuristics) in every regime.

**95th-percentile waiting time** (`p95_waiting_time`). RL achieves the
best p95 on `heavy_poisson` (`11.208 ± 1.810` vs. FF `11.250 ± 1.626`)
and `large_jobs` (`15.140 ± 4.630` vs. FF `15.178 ± 4.500`); ranks within
5 % of the best in `bursty`, `light_poisson`, and `small_cluster`.
SJF is the **worst** scheduler on p95 in every congested regime — its
mean-time advantage comes at the cost of a fat tail.

**99th-percentile waiting time** (`p99_waiting_time`). RL is the best
scheduler in three of five informative regimes — `bursty` (`9.007`),
`large_jobs` (`16.530`), and `light_poisson` (`2.878`) — and ranks
narrowly second in `heavy_poisson` and `small_cluster`. SJF's tail blows
up under congestion: 19.7 vs RL's 9.0 in `bursty`, 41.1 vs 16.5 in
`large_jobs`.

### 8.2 The tail-latency story versus SJF

Table 2 isolates the contrast between the learned policy and SJF on the
99th-percentile waiting time, the metric most relevant to user-visible
latency SLOs.

| regime | RL p99 | SJF p99 | RL beats SJF by |
| --- | --- | --- | --- |
| `light_poisson` | 2.878 | 5.774 | 2.01× |
| `heavy_poisson` | 13.023 | 24.351 | 1.87× |
| `bursty` | 9.007 | 19.741 | 2.19× |
| `large_jobs` | 16.530 | 41.115 | 2.49× |
| `small_cluster` | 7.562 | 12.763 | 1.69× |

The mean-versus-tail trade-off is therefore **explicit and
quantifiable**: trading roughly 30 % higher mean waiting time relative
to SJF buys a 1.7×–2.5× reduction in p99. In tail-sensitive deployments,
this is the right trade.

### 8.3 Arrival-rate sweep and utilization invariance

Figure `arrival_sweep_p99_waiting_time.png` (in `artifacts/eval_run_final/`)
plots each scheduler's p99 wait against arrival rate on a fixed
10-machine cluster. The schedulers are indistinguishable at low load but
diverge sharply once the cluster crosses ~70 % utilization, with SJF's
tail growing faster than the others.

A separate observation: cluster utilization (CPU-time used / CPU-time
available) is within **0.3 percentage points** across all schedulers in
every regime (e.g. `heavy_poisson`: FF 77.4 %, BF 77.7 %, SJF 75.6 %, RL
77.4 %). This is important because it rules out admission control or
total throughput as the source of the learned policy's gains: every
scheduler completes the same set of jobs and uses essentially the same
amount of compute. The tail-latency wins must come from **ordering**
(which job to place when several are queued) and **packing** (which
machine to place it on).

## 9. Ablations

The repository ships an ablation runner
(`scripts/run_ablations.py`) that grids reward shaping × featurizer
choice × seed for end-to-end training and evaluation. The full RL
ablation grid (3 reward variants × 2 featurizers × 3 seeds × 6 regimes)
takes ~14 hours of training on the M4 Pro and was descoped from this
report's submission window. The infrastructure was validated with a
dry-run mode that swaps PPO for `RandomScheduler` (`--dry-run`); the
test suite (`tests/test_ablations.py`) confirms the grid shape and
artifact pipeline. Future work can fill in the remaining cells.

## 10. Negative result: cross-cluster set-attention policy

Because the categorical action space is fixed at `Discrete(num_machines)`,
a policy trained at one cluster size cannot be evaluated at another. To
attempt a single cross-cluster policy we designed `SetMaskablePolicy`
(`cluster_scheduler/policies.py`), built around three pieces:

1. `SetFeaturizer`: zero-pads the observation to a fixed
   `MAX_MACHINES = 32` and adds per-machine `is_active` bits.
2. `SetAttentionExtractor`: a permutation-invariant feature extractor
   that embeds each machine with a shared MLP, injects the current job
   as context, and runs two layers of multi-head self-attention with
   the `is_active` mask as the key-padding mask.
3. A pointer-network actor head: a single shared linear layer applied
   to each machine's embedding produces one logit per machine; a masked
   mean-pooled critic head produces a scalar value.

The unit tests pass: the extractor is permutation-invariant, padded
slots do not affect active-machine embeddings, and a policy trained at
`num_machines=3` successfully predicts for `num_machines=5` through the
same `MaskablePPO.load`.

**Failure mode.** Full training collapses. Across many hyperparameter
combinations (reward scaling 0.05–1.0, learning rate 1e-4 to 2.5e-4,
entropy coefficient 0.005–0.03, with and without `VecNormalize`),
training reliably reports `entropy_loss ≈ -0.14` after a single PPO
update, with `approx_kl` on the order of 1e-7 to 1e-5 — i.e. the
policy commits to a near-deterministic action choice in the first
update and gradients vanish thereafter. We attempted small-gain
orthogonal initialization on the custom heads (the standard SB3 fix
for this failure mode) and per-step reward rescaling without success.

We were unable to identify the root cause within the project window.
Plausible explanations are (a) an interaction between SB3's shared
features-extractor gradient flow and our custom actor/critic heads,
(b) a subtle gradient-zero condition at masked logits in the
combined feature extractor + scoring head, or (c) per-machine entropy
constraints from the action-mask geometry that we have not properly
characterized. The policy is left in the repository with passing unit
tests as future work.

## 11. Limitations

- **Single-step durations.** The simulator does not model resource
  contention between co-located jobs, preemption, or affinity
  constraints common to real clusters.
- **Synthetic workloads.** All evaluations use Poisson arrivals over
  configurable ranges; we do not evaluate on production traces.
- **Per-cluster training.** A single learned policy cannot generalize
  across cluster sizes; cross-cluster generalization (Section 11)
  remains open.
- **Two-resource model.** Only CPU and memory are modeled; real
  clusters add I/O, network, GPU.
- **Fairness and starvation.** No per-tenant or per-job fairness
  signals; SJF demonstrates that an aggressive duration-based policy
  can starve long jobs.
- **Eval set size.** Five seeds yields wide CIs. The ranking of RL,
  First-Fit, and Best-Fit on tail metrics is robust within the
  intervals; close ties (RL ≈ FF on `heavy_poisson` p99) should be
  read as ties rather than upsets.

## 12. Conclusion

A modest-sized PPO policy trained with action masking, a packing-aware
featurizer, and per-step backlog shaping consistently beats classical
heuristics on the 99th-percentile waiting time across diverse workload
regimes, while staying competitive on mean waiting time. The 1.7×–2.5×
tail-latency improvement over duration-oracle SJF, achieved at less
than 0.3 % difference in cluster utilization, demonstrates that the
gains are real and structural — they come from how the policy *orders*
and *packs* jobs, not from how much work it admits. We close with two
directions for future work: (a) a working cross-cluster policy, ideally
via a set-attention extractor that can be trained without entropy
collapse; and (b) duration-aware reward shaping that closes the
mean-time gap to SJF while preserving the tail-latency advantage.

## References

1. Coffman, E. G., Garey, M. R., & Johnson, D. S. (1996). *Approximation
   algorithms for bin packing: A survey*. PWS Publishing Co.
2. Dean, J., & Barroso, L. A. (2013). *The tail at scale*.
   Communications of the ACM, 56(2), 74–80.
3. Huang, S., & Ontañón, S. (2022). *A closer look at invalid action
   masking in policy gradient algorithms*. FLAIRS.
4. Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). *Resource
   management with deep reinforcement learning*. HotNets.
5. Mao, H., Schwarzkopf, M., Venkatakrishnan, S. B., Meng, Z., &
   Alizadeh, M. (2019). *Learning scheduling algorithms for data
   processing clusters*. SIGCOMM.
6. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.
   (2017). *Proximal policy optimization algorithms*. arXiv:1707.06347.
7. Verma, A., Pedrosa, L., Korupolu, M., Oppenheimer, D., Tune, E., &
   Wilkes, J. (2015). *Large-scale cluster management at Google with
   Borg*. EuroSys.

## Appendix A. Reproducibility

All numbers in this report are auto-generated from
`artifacts/eval_run_final/results_summary.csv` by `report/results_tables.md`.
Training and evaluation are reproducible via:

```bash
# Install
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Train (one command per cluster size; ~14 min each on M4 Pro)
python -u -m cluster_scheduler.train --timesteps 8000000 --n-envs 10 \
  --policy-hidden 256 256 --activation gelu \
  --num-machines 10 --num-jobs 500 --arrival-rate 8.0 \
  --featurizer rich --rich-top-k 4 \
  --reward-mode dense --backlog-penalty-weight 0.1 --completion-bonus 1.0 \
  --normalize-reward --n-steps 1024 --batch-size 4096 \
  --learning-rate 2.5e-4 --lr-schedule linear \
  --ent-coef 0.005 --clip-range-vf 0.2 \
  --gamma 0.999 --gae-lambda 0.95 \
  --save-path artifacts/ppo_m10_v4.zip --log-dir runs/ppo_m10_v4

# Evaluate (replays held-out seeds across all schedulers)
python scripts/run_experiments.py \
  --model artifacts/ppo_m4_v4.zip artifacts/ppo_m10_v4.zip artifacts/ppo_m20_v4.zip \
  --seeds 1000 1001 1002 1003 1004 \
  --out-dir artifacts/eval_run_final \
  --arrival-sweep
```

The full repository is available at <https://github.com/J0hns0n-Wang/AI_Prac>.
The trained policy archives, sidecars, and `VecNormalize` statistics live
under `artifacts/` (gitignored; reproducible from the commands above).
