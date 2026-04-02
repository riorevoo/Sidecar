"""APScheduler-based pipeline scheduler.

Runs the full pipeline on a fixed cron schedule.
Prevents overlapping runs using a job lock.
"""
from __future__ import annotations

import logging
import threading

from config import settings

logger = logging.getLogger(__name__)

_run_lock = threading.Lock()


def run_pipeline_job() -> None:
    """Job function invoked by the scheduler."""
    if not _run_lock.acquire(blocking=False):
        logger.warning("Pipeline job skipped — previous run still in progress")
        return

    try:
        logger.info("Scheduled pipeline run starting...")
        from pipeline.runner import PipelineRunner
        runner = PipelineRunner()
        ctx = runner.run()
        logger.info(
            "Scheduled pipeline run complete: %d videos generated",
            len(ctx.videos),
        )
    except Exception as exc:
        logger.exception("Scheduled pipeline run failed: %s", exc)
    finally:
        _run_lock.release()


def start_scheduler() -> None:
    """Start the APScheduler with configured cron expression."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        raise RuntimeError("APScheduler is not installed. Run: pip install APScheduler")

    sched = settings.scheduling
    scheduler = BlockingScheduler(timezone="UTC")

    # Parse cron parts from the expression
    cron_parts = sched.ingest_cron.split()
    if len(cron_parts) == 5:
        minute, hour, day, month, day_of_week = cron_parts
    else:
        minute, hour, day, month, day_of_week = "*/30", "*", "*", "*", "*"

    scheduler.add_job(
        run_pipeline_job,
        CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone="UTC",
        ),
        id="pipeline_job",
        replace_existing=True,
        max_instances=1,
    )

    logger.info(
        "Scheduler started. Pipeline cron: %s (UTC)",
        sched.ingest_cron,
    )

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()
