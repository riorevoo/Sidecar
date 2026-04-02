# Sidecar — AI News Video Pipeline

Fully automated pipeline that fetches global news, groups it into stories, ranks them, writes scripts, synthesizes speech, and generates short-form videos (9:16) ready for TikTok, YouTube Shorts, and Instagram Reels.

```
Ingest → Clean → Embed → Cluster → Enrich → Rank → Script → TTS → Video → Store → Publish
```

---

## Requirements

- Python 3.12
- CUDA-capable GPU (tested on RTX 3070 Ti, 8 GB VRAM)
- FFmpeg installed and on PATH
- An OpenAI API key **or** Ollama running locally

---

## 1. Clone and create a virtual environment

```powershell
cd C:\Users\revaa\IdeaProjects\Sidecar
python -m venv .venv
.venv\Scripts\activate
```

---

## 2. Install PyTorch (do this first, separately)

```powershell
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If pip errors on a `.deleteme` file:

```powershell
# Run this outside the venv, then retry
del C:\Python312\Scripts\tqdm.exe.deleteme
```

---

## 4. Configure environment

```powershell
copy .env.example .env
```

Open `.env` and fill in at minimum:

| Key | What it is | Required |
|---|---|---|
| `OPENAI_API_KEY` | For LLM script generation | Yes (unless using Ollama) |
| `LLM__PROVIDER` | `openai` or `ollama` | Yes |
| `TTS__KOKORO_DEFAULT_VOICE` | Voice for audio synthesis | Yes |

Everything else has sensible defaults. Full reference in `.env.example`.

### Using Ollama instead of OpenAI

1. Download Ollama from [ollama.com](https://ollama.com)
2. Run `ollama pull llama3:8b` (or any model you prefer)
3. Set in `.env`:
   ```
   LLM__PROVIDER=ollama
   LLM__OLLAMA_MODEL=llama3:8b
   ```

---

## 5. Set up TTS — Kokoro (default)

Kokoro is the default TTS provider. It runs locally, requires Python 3.12, and downloads its model (~300 MB) automatically on first use.

Models download on first run — no manual download needed.

Configure voice in `.env`:

```env
TTS__PROVIDER=kokoro
TTS__KOKORO_DEFAULT_VOICE=af_nova
```

Available voices:

| American English (lang=a) | |
|---|---|
| Female | `af_heart` `af_bella` `af_nicole` `af_nova` `af_sky` `af_river` |
| Male | `am_adam` `am_michael` `am_onyx` `am_echo` `am_liam` `am_puck` |

| British English (lang=b) | |
|---|---|
| Female | `bf_emma` `bf_isabella` `bf_alice` |
| Male | `bm_george` `bm_lewis` `bm_daniel` |

Verify it works:

```powershell
python -c "
from kokoro import KPipeline
import numpy as np, soundfile as sf
p = KPipeline(lang_code='a')
audio = np.concatenate([a for _,_,a in p('Testing. This is your AI news anchor.', voice='af_nova')])
sf.write('test_tts.wav', audio, 24000)
print('TTS ok — play test_tts.wav')
"
```

### Tone-based voice control

Scripts are automatically assigned a delivery tone based on story topic. You can optionally set a different voice or speed per tone in `.env`:

```
# Topics → tones:
# crime/breaking          → urgent     (speed 1.1 by default)
# politics/health/world   → serious    (speed 0.92)
# tech/science/business   → analytical (speed 0.95)
# sports/entertainment    → neutral    (speed 1.0)

TTS__KOKORO_VOICE_URGENT=am_onyx
TTS__KOKORO_SPEED_URGENT=1.15
```

---

## 6. Set up video generation

### Option A — Static image (default, no extra setup)

A still image with audio and captions. Fastest, no additional dependencies beyond FFmpeg.

1. Verify FFmpeg: `ffmpeg -version`
2. Drop an anchor face image at `./video/assets/anchor_photo.jpg`
   - Use an AI-generated face from [thispersondoesnotexist.com](https://thispersondoesnotexist.com) (no real person = no likeness issues)
3. Set in `.env`:
   ```env
   VIDEO__PROVIDER=static
   ```

### Option B — SadTalker (animated talking head from a photo)

Animates a still portrait with head movements, blinks, and lip sync. Recommended over static.

```powershell
# Clone the repo
git clone https://github.com/OpenTalker/SadTalker external/SadTalker

# Install its dependencies (in the same venv)
cd external/SadTalker
pip install -r requirements.txt

# Download checkpoints (~2 GB)
bash scripts/download_models.sh
cd ../..
```

Set in `.env`:

```env
VIDEO__PROVIDER=sadtalker
SADTALKER_REPO_PATH=./external/SadTalker
VIDEO__ANCHOR_IMAGE=./video/assets/anchor_photo.jpg
```

### Option C — Wav2Lip (lip-sync only)

Animates only the mouth. Works well if you provide a looping video of a face instead of a still image.

```powershell
git clone https://github.com/Rudrabha/Wav2Lip external/Wav2Lip
# Download wav2lip_gan.pth checkpoint into external/Wav2Lip/checkpoints/
```

```env
VIDEO__PROVIDER=wav2lip
WAV2LIP_REPO_PATH=./external/Wav2Lip
```

---

## 7. Set up news sources

GDELT and RSS work out of the box with no API keys.

```env
INGESTORS=gdelt,rss
RSS_FEEDS=https://feeds.bbci.co.uk/news/rss.xml|https://rss.cnn.com/rss/edition.rss|https://feeds.reuters.com/reuters/topNews
```

To also use NewsAPI (optional, 100 req/day free tier):

```env
INGESTORS=gdelt,rss,newsapi
NEWSAPI_KEY=your_key_here
```

---

## 8. Test each stage individually

Before running the full pipeline, verify each major component:

```powershell
# Test ingestion and cleaning
python main.py test-ingest

# Test LLM script generation with a sample story
python main.py test-script
```

Both commands are read-only — they don't write to the database or generate audio/video.

---

## 9. Database

The database is created automatically on first run — no setup required. It lives at `./data/news_pipeline.db` (configurable via `STORAGE__SQLITE_PATH`).

### Tables

| Table | What's stored |
|---|---|
| `articles` | Every ingested article — URL, title, source, cleaned text, `cluster_id` FK |
| `clusters` | Story groups — topic, summary, importance score, generated script, JSON array of source article URLs |
| `runs` | One row per pipeline execution with per-stage counts and final status |
| `videos` | Generated video file paths, duration, provider — linked to a cluster |
| `publish_records` | Per-platform publish attempts (YouTube, Instagram, TikTok) |

### Scripts and source links

After the script stage, each cluster row holds:

- `script_text` — the full generated script as a string
- `article_urls` — JSON array of every source article URL that fed into the script, e.g.:
  ```json
  ["https://www.bbc.com/news/...", "https://reuters.com/..."]
  ```

Articles are also individually linked back to their cluster via `articles.cluster_id`, so you can always join the two tables for full article text alongside the script.

### Switching to PostgreSQL

```env
STORAGE__PROVIDER=postgres
STORAGE__POSTGRES_URL=postgresql+asyncpg://user:password@localhost:5432/news_pipeline
```

Tables are created automatically on first connect.

---

## 10. Run the pipeline

### Full run (ingest → video → store)

```powershell
python main.py run
```

Outputs land in `./output/videos/`. The database is at `./data/news_pipeline.db`.

### Stop after TTS, do video manually

Useful for reviewing or replacing audio before generating video.

```powershell
python main.py run --tts-only
```

This stops after audio generation and writes a manifest file, e.g.:

```
output/videos/run_1_manifest.json
```

The manifest lists each story's audio path, script, and title. You can edit `audio_path` entries to point at your own recordings, then resume:

```powershell
python main.py resume-video output\videos\run_1_manifest.json
```

### Check run history

```powershell
python main.py status
```

### Run on a schedule (runs forever)

```powershell
python main.py schedule
```

Configure frequency in `.env`:

```env
SCHEDULE__INGEST_CRON=*/30 * * * *   # every 30 minutes
```

---

## 11. Publishing (optional)

Publishing is disabled in MVP mode. To enable:

```env
MVP_MODE=false
PUBLISHERS=youtube,instagram
```

### YouTube

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the YouTube Data API v3
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download as `credentials/youtube_client_secrets.json`
5. On first publish, a browser window will open for OAuth consent

```env
YOUTUBE_CLIENT_SECRETS_FILE=./credentials/youtube_client_secrets.json
YOUTUBE_TOKEN_FILE=./credentials/youtube_token.json
YOUTUBE_DEFAULT_PRIVACY=private
```

### Instagram

```env
INSTAGRAM_USERNAME=your_username
INSTAGRAM_PASSWORD=your_password
```

---

## Project structure

```
Sidecar/
├── main.py                  CLI entry point
├── .env                     Your config (never commit this)
├── .env.example             Config template with all options documented
├── requirements.txt
│
├── pipeline/
│   ├── runner.py            Orchestrates all 8 stages in order
│   └── context.py           Data carrier between stages
│
├── ingestion/               Fetches raw articles (GDELT, RSS, NewsAPI)
├── cleaning/                Normalizes and deduplicates text
├── embedding/               Converts text to vectors (local or OpenAI)
├── clustering/              Groups articles into stories by topic
├── enrichment/              Extracts entities, topic, summary per story
├── ranking/                 Scores and selects top N stories
├── script/                  Generates 30-45s news scripts via LLM
│   └── prompts/             System prompt and user template
├── llm/                     LLM abstraction (OpenAI / Ollama)
├── tts/                     Text-to-speech (Kokoro / ElevenLabs / Piper)
├── video/                   Video generation (static / SadTalker / Wav2Lip)
├── storage/                 SQLite database
├── publishing/              YouTube and Instagram upload
├── scheduling/              APScheduler cron runner
│
├── output/videos/           Generated MP4s and audio
├── data/                    SQLite database file
├── voices/                  Reference audio clips (if using Coqui)
└── video/assets/            anchor_photo.jpg and logo.png
```

---

## Swappable providers

Every major component can be swapped by changing one line in `.env`:

| Component | `.env` key | Options |
|---|---|---|
| News sources | `INGESTORS` | `gdelt`, `rss`, `newsapi` |
| Embeddings | `EMBEDDING__PROVIDER` | `local`, `openai` |
| LLM | `LLM__PROVIDER` | `openai`, `ollama` |
| TTS | `TTS__PROVIDER` | `kokoro`, `elevenlabs`, `fishaudio`, `piper`, `coqui` |
| Video | `VIDEO__PROVIDER` | `static`, `sadtalker`, `wav2lip` |
| Database | `STORAGE__PROVIDER` | `sqlite`, `postgres` |

---

## Legal notes (Australia + international)

- Use an AI-generated face from [thispersondoesnotexist.com](https://thispersondoesnotexist.com) for the anchor — avoids all likeness issues
- Scripts are generated from real ingested articles, not hallucinated — reduces defamation exposure
- Add AI disclosure text to videos before publishing (YouTube policy requires it for synthetic media)
- Kokoro TTS uses preset voices — no voice cloning of real people involved
- Consult a media/IP lawyer before monetising at scale
