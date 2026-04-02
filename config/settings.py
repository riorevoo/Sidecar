"""
Central configuration module.
All settings are read from the .env file (or environment variables).
No other module should call os.getenv() or os.environ directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    ingestors: str = "gdelt,rss"
    newsapi_key: str = ""
    rss_feeds: str = "https://feeds.bbci.co.uk/news/rss.xml"
    ingest_lookback_hours: int = 6
    ingest_max_articles: int = 500

    @property
    def ingestor_list(self) -> list[str]:
        return [i.strip() for i in self.ingestors.split(",") if i.strip()]

    @property
    def rss_feed_list(self) -> list[str]:
        return [f.strip() for f in self.rss_feeds.split("|") if f.strip()]


class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING__", env_file=".env", extra="ignore")

    provider: str = "local"
    local_model: str = "all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    openai_batch_size: int = 100


class ClusteringConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLUSTERING__", env_file=".env", extra="ignore")

    threshold: float = 0.80
    min_cluster_size: int = 2
    max_clusters: int = 50


class EnrichmentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENRICHMENT__", env_file=".env", extra="ignore")

    enabled: bool = False
    spacy_model: str = "en_core_web_sm"


class RankingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RANKING__", env_file=".env", extra="ignore")

    top_n: int = 5
    decay_half_life_hours: float = 12.0
    weight_coverage: float = 0.30
    weight_diversity: float = 0.20
    weight_velocity: float = 0.20
    weight_entity: float = 0.20
    weight_llm: float = 0.10


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM__", env_file=".env", extra="ignore")

    provider: str = "openai"
    openai_model: str = "gpt-4o"
    openai_max_tokens: int = 600
    openai_temperature: float = 0.7
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:8b"
    ollama_max_tokens: int = 600
    ollama_temperature: float = 0.7


class TTSConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS__", env_file=".env", extra="ignore")

    provider: str = "kokoro"

    # Kokoro TTS
    kokoro_lang: str = "a"              # 'a' = American English, 'b' = British English
    kokoro_default_voice: str = "af_nova"
    kokoro_voice_neutral: str = ""
    kokoro_voice_serious: str = ""
    kokoro_voice_urgent: str = ""
    kokoro_voice_analytical: str = ""
    kokoro_speed_neutral: str = ""
    kokoro_speed_serious: str = ""
    kokoro_speed_urgent: str = ""
    kokoro_speed_analytical: str = ""

    # Fish Audio (API-based alternative)
    fishaudio_default_voice: str = ""
    fishaudio_voice_neutral: str = ""
    fishaudio_voice_serious: str = ""
    fishaudio_voice_urgent: str = ""
    fishaudio_voice_analytical: str = ""

    # Piper (kept for local/offline fallback)
    piper_executable: str = "piper"
    piper_model: str = "en_US-lessac-medium"
    piper_models_dir: str = "./models/piper"

    # Coqui XTTS-v2
    coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    coqui_language: str = "en"
    coqui_default_ref_audio: str = ""   # fallback for any tone without a specific clip
    coqui_ref_audio_neutral: str = ""
    coqui_ref_audio_serious: str = ""
    coqui_ref_audio_urgent: str = ""
    coqui_ref_audio_analytical: str = ""

    # ElevenLabs
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_multilingual_v2"


class VideoConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEO__", env_file=".env", extra="ignore")

    provider: str = "static"
    anchor_image: str = "./video/assets/anchor_photo.jpg"
    logo_path: str = "./video/assets/logo.png"
    width: int = 1080
    height: int = 1920
    fps: int = 30
    output_dir: str = "./output/videos"


class StorageConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORAGE__", env_file=".env", extra="ignore")

    provider: str = "sqlite"
    sqlite_path: str = "./data/news_pipeline.db"
    postgres_url: str = ""


class PublishingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    publishers: str = ""
    youtube_client_secrets_file: str = "./credentials/youtube_client_secrets.json"
    youtube_token_file: str = "./credentials/youtube_token.json"
    youtube_default_privacy: str = "private"
    instagram_username: str = ""
    instagram_password: str = ""
    instagram_session_file: str = "./credentials/instagram_session.json"
    tiktok_session_id: str = ""

    @property
    def publisher_list(self) -> list[str]:
        return [p.strip() for p in self.publishers.split(",") if p.strip()]


class SchedulingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCHEDULE__", env_file=".env", extra="ignore")

    ingest_cron: str = "*/30 * * * *"
    pipeline_delay_minutes: int = 2
    publish_cron: str = "0 8,12,17,21 * * *"


class PipelineConfig(BaseSettings):
    """
    Root config object. Reads from .env file.
    Access nested configs as attributes: settings.embedding.provider
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mvp_mode: bool = True
    force_cpu: bool = False
    max_vram_gb: Optional[float] = None

    @field_validator("max_vram_gb", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # Resource safeguard thresholds (set to 0 to disable)
    resource_max_ram_pct: float = Field(default=90.0, description="Abort if system RAM % exceeds this")
    resource_max_cpu_pct: float = Field(default=95.0, description="Abort if CPU % exceeds this")
    resource_check_interval_s: float = Field(default=10.0, description="Seconds between resource checks")
    openai_api_key: str = ""
    elevenlabs_api_key: str = ""
    fishaudio_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    wav2lip_repo_path: str = "./external/Wav2Lip"
    sadtalker_repo_path: str = "./external/SadTalker"

    # Nested configs are instantiated fresh so they each pick up their own env prefixes
    @property
    def ingestion(self) -> IngestionConfig:
        return IngestionConfig()

    @property
    def embedding(self) -> EmbeddingConfig:
        return EmbeddingConfig()

    @property
    def clustering(self) -> ClusteringConfig:
        return ClusteringConfig()

    @property
    def enrichment(self) -> EnrichmentConfig:
        return EnrichmentConfig()

    @property
    def ranking(self) -> RankingConfig:
        return RankingConfig()

    @property
    def llm(self) -> LLMConfig:
        return LLMConfig()

    @property
    def tts(self) -> TTSConfig:
        return TTSConfig()

    @property
    def video(self) -> VideoConfig:
        return VideoConfig()

    @property
    def storage(self) -> StorageConfig:
        return StorageConfig()

    @property
    def publishing(self) -> PublishingConfig:
        return PublishingConfig()

    @property
    def scheduling(self) -> SchedulingConfig:
        return SchedulingConfig()


# Singleton — import this everywhere
settings = PipelineConfig()
