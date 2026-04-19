"""Discrete-event cluster simulator."""

from dataclasses import dataclass, field
from collections import deque

from cluster_scheduler.models import Job, Machine
from cluster_scheduler.scheduler import Scheduler


@dataclass
class SimulatorConfig:
    """Configuration for the cluster simulator.

    Attributes:
        num_machines: Number of machines in the cluster.
        cpu_per_machine: CPU capacity of each machine.
        memory_per_machine: Memory capacity of each machine.
    """

    num_machines: int = 10
    cpu_per_machine: float = 8.0
    memory_per_machine: float = 16.0


class Simulator:
    """Runs a discrete-event simulation of job scheduling on a cluster.

    The simulator advances time event-by-event (job arrivals and completions),
    maintains a wait queue, and delegates placement decisions to a Scheduler.
    """

    def __init__(self, config: SimulatorConfig | None = None):
        self.config = config or SimulatorConfig()
        self.machines: list[Machine] = []
        self.wait_queue: deque[Job] = deque()
        self.completed_jobs: list[Job] = []
        self.current_time: float = 0.0
        self._init_machines()

    def _init_machines(self) -> None:
        cfg = self.config
        self.machines = [
            Machine(
                machine_id=i,
                cpu_capacity=cfg.cpu_per_machine,
                memory_capacity=cfg.memory_per_machine,
            )
            for i in range(cfg.num_machines)
        ]

    def reset(self) -> None:
        """Reset the simulator to its initial state."""
        self.wait_queue.clear()
        self.completed_jobs.clear()
        self.current_time = 0.0
        for m in self.machines:
            m.reset()

    def _release_completed_jobs(self) -> None:
        """Release all jobs that have finished by current_time."""
        for machine in self.machines:
            completed = machine.release_completed_jobs(self.current_time)
            self.completed_jobs.extend(completed)

    def _try_place_from_queue(self, scheduler: Scheduler) -> None:
        """Repeatedly ask the scheduler to place jobs until it can't place any more."""
        while self.wait_queue:
            queue_list = list(self.wait_queue)
            result = scheduler.schedule(queue_list, self.machines)
            if result is None:
                break
            ji, mi = result
            job = queue_list[ji]
            self.machines[mi].place_job(job, self.current_time)
            self.wait_queue.remove(job)

    def _next_completion_time(self) -> float | None:
        """Find the earliest job completion time across all machines."""
        earliest = None
        for machine in self.machines:
            for job in machine.running_jobs:
                if earliest is None or job.completion_time < earliest:
                    earliest = job.completion_time
        return earliest

    def run(self, jobs: list[Job], scheduler: Scheduler) -> list[Job]:
        """Run the full simulation.

        Args:
            jobs: List of jobs sorted by arrival_time.
            scheduler: The scheduler to use for placement decisions.

        Returns:
            List of all completed jobs with timing information filled in.
        """
        self.reset()
        job_index = 0
        total_jobs = len(jobs)

        while job_index < total_jobs or self.wait_queue:
            # Determine next event time: either a job arrival or a job completion
            next_arrival = jobs[job_index].arrival_time if job_index < total_jobs else None
            next_completion = self._next_completion_time()

            # Pick the earlier event
            if next_arrival is not None and (next_completion is None or next_arrival <= next_completion):
                # Advance to next arrival
                self.current_time = next_arrival

                # Release any jobs that completed by now
                self._release_completed_jobs()

                # Try to place queued jobs first (they've been waiting longer)
                self._try_place_from_queue(scheduler)

                # Add newly arrived job to the queue, then let scheduler decide
                job = jobs[job_index]
                job_index += 1
                self.wait_queue.append(job)
                self._try_place_from_queue(scheduler)

            elif next_completion is not None:
                # Advance to next completion
                self.current_time = next_completion
                self._release_completed_jobs()
                self._try_place_from_queue(scheduler)

            else:
                # No more events — should not happen if logic is correct
                break

        # Final: advance to last completion
        while self._next_completion_time() is not None:
            self.current_time = self._next_completion_time()
            self._release_completed_jobs()

        return list(self.completed_jobs)
