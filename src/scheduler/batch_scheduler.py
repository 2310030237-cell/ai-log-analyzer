"""
Batch Scheduler
Orchestrates the full batch processing pipeline at scheduled intervals.
"""

import time
import schedule
from datetime import datetime
from typing import Callable, Optional


class BatchScheduler:
    """
    Schedules and orchestrates batch processing jobs.
    Uses the 'schedule' library for periodic execution.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        sched_config = self.config.get("scheduler", {})
        self.interval = sched_config.get("interval", "daily")
        self.time_str = sched_config.get("time", "02:00")
        self.job_fn = None
        self.is_running = False

    def set_job(self, job_fn: Callable):
        """Set the job function to execute on schedule."""
        self.job_fn = job_fn

    def _run_job(self):
        """Execute the scheduled job with logging."""
        print(f"\n{'='*60}")
        print(f"SCHEDULED JOB STARTED: {datetime.now().isoformat()}")
        print(f"{'='*60}")

        try:
            if self.job_fn:
                self.job_fn()
            print(f"\n[OK] Scheduled job completed at {datetime.now().isoformat()}")
        except Exception as e:
            print(f"\n[X] Scheduled job FAILED: {e}")

    def start(self):
        """Start the scheduler."""
        if self.job_fn is None:
            raise RuntimeError("No job function set. Call set_job() first.")

        print(f"\n{'='*60}")
        print(f"BATCH SCHEDULER")
        print(f"{'='*60}")
        print(f"  Interval: {self.interval}")
        print(f"  Time: {self.time_str}")
        print(f"  Press Ctrl+C to stop\n")

        # Configure schedule
        if self.interval == "daily":
            schedule.every().day.at(self.time_str).do(self._run_job)
        elif self.interval == "weekly":
            schedule.every().monday.at(self.time_str).do(self._run_job)
        elif self.interval == "hourly":
            schedule.every().hour.do(self._run_job)
        elif self.interval == "minute":  # For testing
            schedule.every(1).minutes.do(self._run_job)
        else:
            print(f"  Unknown interval '{self.interval}', defaulting to daily")
            schedule.every().day.at(self.time_str).do(self._run_job)

        self.is_running = True
        next_run = schedule.next_run()
        print(f"  Next run: {next_run}")
        print(f"  Scheduler active. Waiting...\n")

        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n  Scheduler stopped by user.")
            self.is_running = False

    def run_once(self):
        """Run the job immediately (manual trigger)."""
        self._run_job()

    def stop(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        print("  Scheduler stopped.")
