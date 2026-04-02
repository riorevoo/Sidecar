"""Pipeline runner — orchestrates all stages in order.

This is the most important file in the project. It:
1. Instantiates all stage components via their factories
2. Passes PipelineContext through each stage
3. Handles per-stage errors (transient vs permanent)
4. Writes results to storage
5. Respects MVP_MODE to skip non-core stages

stop_after_tts mode:
  run(stop_after_tts=True) halts after TTS and writes a manifest JSON so
  you can do the video step manually, then resume with resume_from_manifest().
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from rich.console import Console

from config import settings
from .context import PipelineContext

logger = logging.getLogger(__name__)
console = Console()


class PipelineRunner:
    """Runs the full news video pipeline end-to-end."""

    def __init__(self) -> None:
        self._cfg = settings
        self._storage = None
        self._embedder = None
        self._tts = None
        self._video_gen = None

    def _get_storage(self):
        if self._storage is None:
            from storage.factory import get_storage
            self._storage = get_storage()
        return self._storage

    def run(self, stop_after_tts: bool = False) -> PipelineContext:
        """Execute the pipeline and return the context with all results.

        Args:
            stop_after_tts: If True, halt after TTS and write a manifest file.
                            Call resume_from_manifest() later to do the video step.
        """
        from utils.hardware import ResourceGuard, ResourceLimitExceeded

        guard: ResourceGuard | None = None
        if self._cfg.resource_max_ram_pct > 0 or self._cfg.resource_max_cpu_pct > 0:
            guard = ResourceGuard(
                max_ram_pct=self._cfg.resource_max_ram_pct,
                max_cpu_pct=self._cfg.resource_max_cpu_pct,
                interval_s=self._cfg.resource_check_interval_s,
            )
            guard.start()

        ctx = PipelineContext()

        storage = self._get_storage()
        ctx.run_id = storage.create_run()
        logger.info("Pipeline run %d started", ctx.run_id)

        def _check() -> None:
            if guard:
                guard.check()

        status = "success"
        try:
            self._stage_ingest(ctx);   _check()
            self._stage_clean(ctx);    _check()
            self._stage_embed_and_cluster(ctx); _check()
            self._stage_enrich(ctx);   _check()
            self._stage_rank(ctx);     _check()
            self._stage_script(ctx);   _check()
            self._stage_tts(ctx);      _check()

            if stop_after_tts:
                # Save what we have so far so resume can read it back
                storage.save_clusters(ctx.ranked_clusters, ctx.run_id)
                manifest_path = self._write_manifest(ctx)
                storage.complete_run(
                    ctx.run_id,
                    status="paused_after_tts",
                    articles_ingested=len(ctx.raw_articles),
                    articles_cleaned=len(ctx.clean_articles),
                    clusters_formed=len(ctx.clusters),
                    clusters_ranked=len(ctx.ranked_clusters),
                    scripts_generated=sum(1 for c in ctx.ranked_clusters if c.script),
                    videos_generated=0,
                )
                console.print(f"\n[bold yellow]Stopped after TTS.[/bold yellow]")
                console.print(f"Manifest written to: [cyan]{manifest_path}[/cyan]")
                console.print("Review / replace the audio files, then run:")
                console.print(f"  [bold]python main.py resume-video --manifest {manifest_path}[/bold]")
                self._log_funnel(ctx)
                return ctx

            self._stage_video(ctx)
            self._stage_store_results(ctx)

            if not self._cfg.mvp_mode:
                self._stage_publish(ctx)

        except ResourceLimitExceeded as exc:
            logger.critical("Pipeline run %d aborted — resource limit: %s", ctx.run_id, exc)
            console.print(f"\n[bold red]Aborted:[/bold red] {exc}")
            ctx.record_error("resource_limit", exc)
            status = "failed"

        except Exception as exc:
            logger.exception("Pipeline run %d failed: %s", ctx.run_id, exc)
            ctx.record_error("pipeline", exc)
            status = "failed"

        finally:
            if guard:
                guard.stop()
            if status != "paused_after_tts":
                storage.complete_run(
                    ctx.run_id,
                    status=status,
                    articles_ingested=len(ctx.raw_articles),
                    articles_cleaned=len(ctx.clean_articles),
                    clusters_formed=len(ctx.clusters),
                    clusters_ranked=len(ctx.ranked_clusters),
                    scripts_generated=sum(1 for c in ctx.ranked_clusters if c.script),
                    videos_generated=len(ctx.videos),
                )
                self._log_funnel(ctx)

        return ctx

    def resume_from_manifest(self, manifest_path: str) -> PipelineContext:
        """Pick up from a saved TTS manifest and run video + storage.

        Each entry in the manifest has:
          cluster_id, script_text, audio_path, story_title

        You can edit audio_path entries before resuming to point at your own files.
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        console.print(f"[bold cyan]Resuming from manifest:[/bold cyan] {manifest_path}")
        console.print(f"  → {len(entries)} stories to process")

        ctx = PipelineContext()
        storage = self._get_storage()
        ctx.run_id = storage.create_run()

        # Reconstruct just enough context for the video stage
        from tts.models import AudioResult
        from script.models import Script
        from clustering.models import Cluster

        for entry in entries:
            audio_path = entry["audio_path"]
            if not Path(audio_path).exists():
                console.print(f"  [yellow]⚠ Audio not found, skipping: {audio_path}[/yellow]")
                continue

            script = Script(
                cluster_id=entry["cluster_id"],
                full_text=entry["script_text"],
            )
            audio = AudioResult(
                output_path=audio_path,
                duration_seconds=entry.get("estimated_duration_seconds", 0.0),
                story_id=entry["cluster_id"],
            )

            # Attach to a minimal cluster so _stage_video can iterate
            cluster = Cluster(cluster_id=entry["cluster_id"])
            cluster.script = script
            cluster._resume_audio = audio  # stash audio for _stage_video
            ctx.ranked_clusters.append(cluster)

        self._stage_video(ctx, from_manifest=True)
        self._stage_store_results(ctx)

        storage.complete_run(
            ctx.run_id,
            status="success",
            videos_generated=len(ctx.videos),
        )
        self._log_funnel(ctx)
        return ctx

    # ──────────────────────────────────────────────────────────
    #  Stage implementations
    # ──────────────────────────────────────────────────────────

    def _stage_ingest(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 1/8: Ingestion[/bold cyan]")
        from ingestion.factory import build_ingestors

        ingestors = build_ingestors()
        all_articles = []

        for ingestor in ingestors:
            try:
                articles = asyncio.run(ingestor.fetch())
                all_articles.extend(articles)
                logger.info("[%s] Fetched %d articles", ingestor.name, len(articles))
            except Exception as exc:
                logger.warning("Ingestor '%s' failed: %s", ingestor.name, exc)
                ctx.record_error(f"ingest:{ingestor.name}", exc)

        ctx.raw_articles = all_articles
        console.print(f"  → {len(all_articles)} raw articles fetched")

    def _stage_clean(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 2/8: Cleaning[/bold cyan]")
        from cleaning.cleaner import TextCleaner

        cleaner = TextCleaner()
        ctx.clean_articles = cleaner.clean(ctx.raw_articles)
        console.print(f"  → {len(ctx.clean_articles)} articles after cleaning")

        if not ctx.clean_articles:
            raise RuntimeError("No articles survived cleaning — aborting pipeline")

        # Save to storage (deduplication against DB happens here)
        self._get_storage().save_articles(ctx.clean_articles, ctx.run_id)

    def _stage_embed_and_cluster(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 3/8: Embedding + Clustering[/bold cyan]")
        from embedding.factory import get_embedder
        from clustering.clusterer import CosineClustering

        embedder = get_embedder()
        texts = [a.cleaned_text for a in ctx.clean_articles]

        console.print(f"  → Embedding {len(texts)} texts...")
        embeddings = embedder.embed(texts)

        clusterer = CosineClustering()
        ctx.clusters = clusterer.cluster(ctx.clean_articles, embeddings)
        console.print(f"  → {len(ctx.clusters)} clusters formed")

        if not ctx.clusters:
            raise RuntimeError("No clusters formed — aborting pipeline")

    def _stage_enrich(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 4/8: Enrichment[/bold cyan]")
        from enrichment.factory import build_enricher

        enricher = build_enricher()
        ctx.clusters = enricher.enrich(ctx.clusters)
        console.print(f"  → {len(ctx.clusters)} clusters enriched")

    def _stage_rank(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 5/8: Ranking[/bold cyan]")
        from ranking.ranker import ClusterRanker

        ranker = ClusterRanker()
        ctx.ranked_clusters = ranker.rank(ctx.clusters)
        console.print(f"  → Top {len(ctx.ranked_clusters)} stories selected")

    def _stage_script(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 6/8: Script Generation[/bold cyan]")
        from script.generator import ScriptGenerator

        generator = ScriptGenerator()
        generator.generate_batch(ctx.ranked_clusters)
        generated = sum(1 for c in ctx.ranked_clusters if c.script)
        console.print(f"  → {generated} scripts generated")

    def _stage_tts(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 7a/8: TTS[/bold cyan]")
        from tts.factory import get_tts

        tts = get_tts()
        output_dir = Path(settings.video.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, cluster in enumerate(ctx.ranked_clusters, start=1):
            if not cluster.script:
                continue
            try:
                stem = f"{ctx.run_timestamp}_{i:03d}"
                script_path = output_dir / f"script_{stem}.txt"
                script_path.write_text(cluster.script.full_text, encoding="utf-8")
                console.print(f"  ✓ Script: {script_path}")

                audio_path = str(output_dir / f"audio_{stem}.wav")
                audio_result = tts.synthesize(cluster.script, audio_path)
                cluster._tts_audio = audio_result  # stash for video stage
                console.print(f"  ✓ Audio:  {audio_path}")
            except Exception as exc:
                logger.error("TTS failed for cluster %s: %s", cluster.cluster_id[:8], exc)
                ctx.record_error(f"tts:{cluster.cluster_id[:8]}", exc)

    def _stage_video(self, ctx: PipelineContext, from_manifest: bool = False) -> None:
        console.print("[bold cyan]Stage 7b/8: Video[/bold cyan]")
        from video.factory import get_video_generator

        video_gen = get_video_generator()
        output_dir = Path(settings.video.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, cluster in enumerate(ctx.ranked_clusters, start=1):
            if not cluster.script:
                continue
            # When resuming from manifest the audio is stored as _resume_audio
            audio_result = getattr(cluster, "_resume_audio", None) or getattr(cluster, "_tts_audio", None)
            if audio_result is None:
                logger.warning("No audio for cluster %s — skipping video", cluster.cluster_id[:8])
                continue
            try:
                stem = f"{ctx.run_timestamp}_{i:03d}"
                video_path = str(output_dir / f"video_{stem}.mp4")
                video_result = video_gen.generate(audio_result, cluster.script, video_path)
                ctx.videos.append(video_result)
                console.print(f"  ✓ Video: {video_path}")
            except Exception as exc:
                logger.error("Video failed for cluster %s: %s", cluster.cluster_id[:8], exc)
                ctx.record_error(f"video:{cluster.cluster_id[:8]}", exc)

    def _write_manifest(self, ctx: PipelineContext) -> str:
        """Write a JSON manifest of TTS outputs so the video step can resume later."""
        output_dir = Path(settings.video.output_dir)
        manifest_path = output_dir / f"run_{ctx.run_id}_manifest.json"

        entries = []
        for cluster in ctx.ranked_clusters:
            audio = getattr(cluster, "_tts_audio", None)
            if not audio or not cluster.script:
                continue
            entries.append({
                "cluster_id": cluster.cluster_id,
                "story_title": cluster.representative_title,
                "script_text": cluster.script.full_text,
                "audio_path": audio.output_path,
                "estimated_duration_seconds": audio.duration_seconds,
                "sources": [
                    {"url": a.url, "title": a.title, "source": a.source}
                    for a in cluster.articles
                ],
            })

        manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return str(manifest_path)

    def _stage_store_results(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Stage 8/8: Storing Results[/bold cyan]")
        storage = self._get_storage()

        # Save clusters with scripts
        storage.save_clusters(ctx.ranked_clusters, ctx.run_id)

        # Save video records
        for i, video in enumerate(ctx.videos):
            if i < len(ctx.ranked_clusters):
                storage.save_video(ctx.ranked_clusters[i].cluster_id, video)

        console.print(f"  → Saved {len(ctx.videos)} video records")

    def _stage_publish(self, ctx: PipelineContext) -> None:
        console.print("[bold cyan]Publishing[/bold cyan]")
        from publishing.factory import build_publishers

        publishers = build_publishers()
        if not publishers:
            logger.info("No publishers configured — skipping")
            return

        storage = self._get_storage()
        unpublished = storage.get_unpublished_videos()

        for video_record in unpublished:
            for publisher in publishers:
                try:
                    cluster_id = video_record.cluster_id
                    # Build title/description from stored cluster data
                    result = publisher.publish(
                        video_path=video_record.video_path,
                        title="Latest News Update",
                        description="",
                        tags=["news", "shorts"],
                        thumbnail_path=video_record.thumbnail_path,
                    )
                    if result.success:
                        storage.mark_published(
                            video_record.id,
                            result.platform,
                            result.platform_video_id,
                            result.platform_url,
                        )
                        console.print(f"  ✓ Published to {result.platform}: {result.platform_url}")
                except Exception as exc:
                    logger.error("Publish failed [%s]: %s", publisher.platform_name, exc)

    def _log_funnel(self, ctx: PipelineContext) -> None:
        console.print("\n[bold green]Pipeline Complete — Funnel Summary[/bold green]")
        console.print(f"  Ingested:   {len(ctx.raw_articles)} articles")
        console.print(f"  Cleaned:    {len(ctx.clean_articles)} articles")
        console.print(f"  Clustered:  {len(ctx.clusters)} stories")
        console.print(f"  Ranked:     {len(ctx.ranked_clusters)} top stories")
        console.print(f"  Videos:     {len(ctx.videos)} generated")
        if ctx.errors:
            console.print(f"  [yellow]Errors:     {len(ctx.errors)}[/yellow]")
            for err in ctx.errors:
                console.print(f"    • {err}")
