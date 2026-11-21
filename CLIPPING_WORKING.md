# ExtremeCuts: Complete Architectural & Technical Engine Specification

> **A Complete Engineering Blueprint for Autonomous & Local Video Clipping**  
> *This document provides a comprehensive, self-contained implementation specification of the NeonClipping engine. Any developer or AI agent reading this single document can reconstruct, run, and customize the entire pipeline seamlessly in any new or existing project.*

---

## 1. System Overview & Core Philosophy

NeonClipping is a **100% local, privacy-first, zero-subscription video clipping engine**. It converts long horizontal (16:9) gaming streams, podcasts, and video VODs (YouTube, Twitch, Kick, or local MP4 files) into vertical (9:16, 1080×1920) viral shorts for TikTok, Instagram Reels, and YouTube Shorts.

### Core Architectural Pillars
1. **Zero Cloud Dependencies**: All transcription, computer vision, AI evaluation, and video rendering happen strictly on the local machine
2. **Audio-First Optimization ("Fast Mode")**: Processing an 85-minute 1080p video by downloading and decoding the full video takes massive bandwidth and time. NeonClipping downloads only the audio track (~80 MB) and fetches YouTube's native word-timed auto-captions in ~1 second. Only the selected candidate clips (e.g., 30–60s) are subsequently downloaded.
3. **Resolution-Independent Geometry**: All crop regions (facecam and gameplay) are normalized in the `[0.0, 1.0]` floating-point coordinate space. The same crop geometry works whether the source is 720p, 1080p, or 4K.
4. **Single-Pass Hardware Acceleration**: FFmpeg constructs a complex filter graph that crops, scales, pads, overlays multiple layers, and burns styled ASS subtitles in a single NVENC hardware-accelerated pass (`h264_nvenc`).
5. **Human-in-the-Loop Feedback Learning**: User ratings (1–5 stars) incrementally re-fit feature weights using ridge regression, tailoring highlight detection to personal taste over time.

---

## 2. Technology Stack & Prerequisites

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend Framework** | Python 3.10+ / FastAPI / Uvicorn | Async web server, job management, background threads |
| **Media Downloader** | `yt-dlp` | Stream scraping, thumbnail extraction, cookie authentication |
| **Video Processing** | FFmpeg 6.0+ / `ffprobe` | Demuxing, decoding, audio extraction, filtergraphs, NVENC encoding |
| **Computer Vision** | OpenCV (`opencv-python-headless`) / NumPy | Optical flow (Farneback), cut detection, luminance flash analysis |
| **Speech-to-Text** | `faster-whisper` (CTranslate2) | Word-level timestamped transcription, CUDA int8/float16 execution |
| **Local LLM Engine** | Ollama (`qwen3.5:4b` or `llama3.2:3b`) | Semantic evaluation, viral hook generation, punchy title drafting |
| **Frontend** | Vanilla HTML5 / Canvas / CSS3 / ES6 JS | Dual-canvas interactive cropping, 60fps real-time preview composite |

---

## 3. End-to-End Pipeline Architecture

```
                       ┌─────────────────────────────────────────┐
                       │           Input Source Video            │
                       │  (YouTube / Twitch / Kick / Local MP4)  │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │          Phase 1: Ingestion             │
                       │ - Parse URL & check local cookies.txt   │
                       │ - Background thread: 4 preview frames   │
                       │ - Fast Mode: Audio + YT Captions JSON3  │
                       │ - Deep Mode: 720p MP4 download          │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │      Phase 2: Multi-Signal Scoring      │
                       │ - Audio: 8kHz mono RMS + 2-4kHz band    │
                       │ - 120s rolling z-score baseline         │
                       │ - Vision: Farneback Optical Flow (6fps) │
                       │ - Cuts, Flash bursts, Police strobes    │
                       │ - Excitement Lexicon (EN/HI/Hinglish)   │
                       │ - Candidate window clustering (NMS)     │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │      Phase 3: AI Semantic Ranking       │
                       │ - Read transcript for each candidate    │
                       │ - Sentence snap (end on . ? ! … ।)      │
                       │ - Ollama LLM prompt (Qwen 3.5 / Llama)  │
                       │ - Generates: Score (1-10), Title, Hook  │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │   Phase 4: Layout & Subtitle Engine     │
                       │ - Apply chosen layout (1 of 11 specs)   │
                       │ - Normalize face/game bounding boxes    │
                       │ - Word grouping (2-4 words per group)   │
                       │ - Generate ASS file (Styles: GTA/Beast) │
                       │ - Hindi to Hinglish transliteration     │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │       Phase 5: Single-Pass FFmpeg       │
                       │ - split into stream paths               │
                       │ - crop + scale (aspect ratio increase)  │
                       │ - pad to 1080x1920 + overlay layers     │
                       │ - burn ASS subtitles via libass         │
                       │ - Encode: h264_nvenc (or libx264)       │
                       └────────────────────┬────────────────────┘
                                            │
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │     Output: 1080x1920 Vertical MP4      │
                       │   + SRT Subtitles + Adaptive Feedback   │
                       └─────────────────────────────────────────┘
```

---

## 4. Phase-by-Phase Technical Implementation

### Phase 1: Ingestion & YouTube Anti-Bot Authentication

#### 1. Stream & URL Validation
URLs are strictly validated to prevent SSRF and unsafe protocols:
```python
def allowed(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or '').lower()
    return (
        parsed.scheme in ('http', 'https')
        and not parsed.username
        and not parsed.password
        and host in ('youtu.be', 'youtube.com', 'www.youtube.com', 'm.youtube.com',
                     'twitch.tv', 'www.twitch.tv', 'clips.twitch.tv', 'kick.com', 'www.kick.com')
        and parsed.port in (None, 80, 443)
    )
```

#### 2. YouTube BotGuard & Cookie Resolution
YouTube frequently blocks server environments with `Sign in to confirm you’re not a bot`. NeonClipping resolves this by probing for standard Netscape cookies in priority order:
1. `cookies.txt` (or `youtube_cookies.txt`) in project root.
2. `cookies.txt` in the `data/` directory.
3. Environment variables: `NEONCLIPPING_COOKIES` or `YTDLP_COOKIES`.

```python
def get_cookie_options(data_dir: Path) -> dict:
    candidates = [
        data_dir.parent / 'cookies.txt',
        data_dir.parent / 'youtube_cookies.txt',
        data_dir / 'cookies.txt'
    ]
    env_cookie = os.environ.get('NEONCLIPPING_COOKIES') or os.environ.get('YTDLP_COOKIES')
    if env_cookie and Path(env_cookie).exists():
        return {'cookiefile': str(env_cookie)}
    for candidate in candidates:
        if candidate.exists():
            return {'cookiefile': str(candidate)}
    return {}
```

#### 3. Early Preview Frame Streaming
Users need immediate visual feedback to set up crops without waiting for the entire video to download:
- Downloader queries storyboard URLs from `yt-dlp` info dictionary.
- If storyboard sheets exist, FFmpeg slices 4 evenly spaced thumbnail frames (`0.jpg`, `1.jpg`, `2.jpg`, `3.jpg`) at 15%, 35%, 60%, and 85% video duration.
- The browser fetches `/preview/{job}/{index}` immediately.

---

### Phase 2: Multi-Signal Highlight Detection Engine

Highlight candidates are selected by analyzing synchronized multi-modal signals over the entire duration.

#### 1. Audio Energy & Screaming Detection (`highlights.py`)
Decodes audio to raw 8 kHz, 16-bit mono PCM via FFmpeg pipe:
```bash
ffmpeg -v error -i <source> -vn -ac 1 -ar 8000 -f s16le pipe:1
```
- **Loudness (RMS)**: Sliced into 1-second chunks (8,000 samples per second):
  $$\text{RMS}_t = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}$$
- **Voice Register / Scream Detection (2000–4000 Hz)**:
  Applies Hann window and real FFT (`np.fft.rfft(x * np.hanning(8000))`). Calculates the fraction of total spectral energy situated between 2 kHz and 4 kHz:
  $$\text{HF Ratio}_t = \frac{\sum_{f=2000}^{4000} |X(f)|^2}{\sum_{f=0}^{4000} |X(f)|^2 + \epsilon}$$
- **Rolling Local Z-Score Baseline**:
  A sudden shout in a quiet dialogue scene is far more important than steady loud background music. Signals are standardized against a **120-second rolling window**:
  $$\mu_t = \text{mean}(x_{t-60 \dots t+60}), \quad \sigma_t = \text{std}(x_{t-60 \dots t+60})$$
  $$\text{Score}_t = \text{clip}\left(\frac{x_t - \mu_t}{3.5 \cdot \sigma_t}, 0, 1\right)$$

#### 2. Computer Vision Optical Flow & Action Analysis (`action.py`)
Decodes video chunks at low resolution ($160 \times 90$, 6 fps) directly into raw RGB pipes:
```bash
ffmpeg -v error -ss <start> -t <dur> -i <source> -an -vf fps=6,scale=160:90 -pix_fmt rgb24 -f rawvideo pipe:1
```
- **Scene Cut Detection**: If mean pixel difference between adjacent grayscale frames exceeds $38.0$, it is registered as a hard scene cut (`CUT = 38.0`) and excluded from motion calculation.
- **Motion Magnitude**: Uses **Farneback Optical Flow** (`cv2.calcOpticalFlowFarneback`). Calculates the 90th percentile of velocity vectors $\sqrt{u^2 + v^2}$, normalized such that 8 pixels/frame equals maximum motion ($1.0$).
- **Flash & Explosion Detection**: Sudden luminance rises ($\Delta L > 28$) immediately followed by a drop ($\Delta L < -10$), plus explosions identified via fire-orange color masks:
  $$(R > 200) \land (90 < G < 190) \land (B < 90)$$
- **Police Light Strobe**: Measures high-frequency alternation between dominant red frames ($R > 170, G < 90, B < 90$) and blue frames ($B > 170, R < 90, G < 140$).

#### 3. Excitement Lexicon Signal (`excite.py`)
Evaluates transcripts for viral streamer phrases in English, Hindi, and Hinglish using weighted regular expressions:
- **Tier 3 (Panic/Disbelief, weight = 3)**: `"oh my god"`, `"omg"`, `"no way"`, `"what the fuck"`, `"holy shit"`.
- **Tier 2 (Hype/Action/Hindi, weight = 2)**: `"let's go"`, `"insane"`, `"bhaag"`, `"bhago"`, `"arre yaar"`, `"dhamaka"`, `"police"`, `"headshot"`.
- Matches are aggregated per second, convolved with a 5-second smoothing filter, and blended into the candidate priority score.

#### 4. Non-Maximum Suppression (Clustering)
Candidates are sorted by combined activity score. A greedy suppression window iterates through peaks: if a proposed candidate overlaps an already selected interval $[start, end]$, it is discarded.

---

### Phase 3: AI Semantic Ranking & Snapping

#### 1. Sentence Boundary Snapping (`smart.py`)
Raw candidate windows extracted from audio/motion often cut people off mid-word. The `snap()` algorithm expands or trims the window to grammatical sentence boundaries:
```python
def snap(c, words, min_len, max_len):
    if not words: return
    # If first spoken word starts late, advance start time to right before speech
    lead = words[0]['start']
    if lead > 2.5:
        c['start'] = round(c['start'] + lead - 0.4, 2)
        for w in words:
            w['start'] = round(w['start'] - lead + 0.4, 3)
            w['end'] = round(w['end'] - lead + 0.4, 3)
            
    limit = c['end'] - c['start']
    # Look for terminal punctuation: . ? ! … or Devanagari danda ।
    good = [w['end'] for w in words if w['end'] <= min(limit, max_len) 
            and w['text'].rstrip()[-1:] in '.?!…।' and w['end'] >= min_len]
    ends = [w['end'] for w in words if w['end'] <= min(limit, max_len) and w['end'] >= min_len]
    pick = good[-1] if good else (ends[-1] if ends else None)
    c['end'] = round(c['start'] + (pick + 0.3 if pick else min(limit, max_len)), 2)
```

#### 2. Local Ollama LLM Evaluation
Connects to local Ollama daemon (`http://127.0.0.1:11434/api/generate`) targeting models in order: `qwen3.5:4b`, `llama3.2:3b`, `qwen2.5:3b`.
- **System Prompt**:
  ```
  You are an expert short-form video editor for TikTok, YouTube Shorts, and Reels.
  Given a transcript, rate its viral potential (1-10), give it a short punchy title (max 7 words),
  and write an opening viral hook line (max 8 words).
  Reply ONLY with JSON: {"score": <number 1-10>, "title": "<string>", "hook": "<string>"}
  ```
- **Resilient JSON Parser**: Extracts embedded JSON blocks using regular expressions even if the LLM surrounds output with conversational markdown text.

---

### Phase 4: Subtitle Generation & Transliteration

#### 1. Instant YouTube Transcripts (`captions_yt.py`)
Downloads YouTube's native `json3` timed-text track directly:
- Extracts words, start times (`tStartMs`), and duration offsets (`tOffsetMs`).
- Filters out non-speech tags such as `[Music]`, `[Applause]`, `>>`.
- Yields accurate word-level timings in under 1 second without touching a GPU model.

#### 2. Local Whisper ASR Fallback (`captions.py`)
When YouTube captions do not exist:
- Uses `faster-whisper` (`WhisperModel`) with model `large-v3-turbo` (or `large-v3` for Hindi when $\ge 3.4\text{ GB}$ VRAM is free).
- Runs with `compute_type='int8_float16'` on CUDA, falling back to CPU `int8`.
- Enables `vad_filter=True` (Voice Activity Detection) and `word_timestamps=True`.

#### 3. Hindi-to-Hinglish Transliteration Engine (`translit.py`)
Indian gaming creators frequently speak Hindi, but captions perform significantly better when written in Roman/English letters (Hinglish):
- Maps Devanagari vowels, matras, consonants, conjuncts, and nuktas into phonetically natural Roman equivalents (`"नमस्ते"` $\rightarrow$ `"namaste"`, `"भाई"` $\rightarrow$ `"bhai"`).
- Handles implicit schwa deletion (halant rules) to ensure fluent reading.

#### 4. Word Grouping & ASS Subtitle Generation
Subtitles are grouped into rhythmic 2–4 word bursts ([captions.py](file:///d:/New%20folder/neonclipping-main/captions.py#L96-L111)):
- **Group Break Conditions**: Punctuation (`. ? ! , : ; । …`), pauses $> 0.45\text{ s}$, max words ($4$), or max characters ($24$).
- **Color Themes & ASS Tags**:
  - `gta`: Neon Cyan (`&H00EFFF43&`) resting text with hot Pink (`&H00A845FF&`) active word glow.
  - `mrbeast`: Giant Yellow/Orange font with heavy 9px outline and active-word pop scaling ($112\%$).
  - `hormozi`: White text with vibrant Cyan keyword pop and drop shadow.
  - `clean`: Minimalist semi-bold white subtitle at base.
- Generates precise ASS dialogue events with karaoke/active timing tags:
  ```
  Dialogue: 1,0:00:12.10,0:00:12.45,Main,,0,0,0,,{\pos(540,1450)\fs72\bord6}THIS IS {\c&H00A845FF&}WILD
  ```

---

### Phase 5: Spatial Framing & Layout Geometry

#### 1. The 11 Layout Geometries (`layouts.py`)
Target resolution is strictly vertical **$1080 \times 1920$** ($9:16$ aspect ratio).
Let $top \in [0.20, 0.50]$ be the facecam height ratio (default $0.32$). The pixel height of the facecam is $h = \text{int}(top \times 1920)$.

| Layout ID | Name | Geometric Architecture |
| :--- | :--- | :--- |
| `stack` | Face + Gameplay | Camera at top ($y=0, h=614\text{px}$), gameplay below ($y=614, h=1306\text{px}$). |
| `bottom` | Gameplay + Face | Gameplay at top ($y=0, h=1306\text{px}$), camera below ($y=1306, h=614\text{px}$). |
| `blur` | Blurred Fill | 16:9 source centered; background is $16\text{px}$ box-blurred zoom filling $1080\times 1920$. |
| `fill` | Fill Screen | Full-frame center-crop scaled to fill $1080\times 1920$. |
| `game` | Gameplay Only | Only the selected gameplay box scaled to fill $1080\times 1920$. |
| `pip_tl` | Top-Left PIP | Fullscreen gameplay background; floating camera box ($w=int(top \times 1080)$) at top-left. |
| `pip_tr` | Top-Right PIP | Fullscreen gameplay background; floating camera box at top-right. |
| `pip_bl` | Bottom-Left PIP | Fullscreen gameplay background; floating camera box at bottom-left ($y=1920-h-180$). |
| `pip_br` | Bottom-Right PIP | Fullscreen gameplay background; floating camera box at bottom-right. |
| `circle` | Circle Camera | Round facecam mask placed above gameplay using RGBA alpha circular formula. |
| `side` | Side by Side | Left column facecam ($w = top \times 1080$), right column gameplay ($1080 - w$). |

#### 2. The Browser Dual-Canvas Engine (`app.js`)
- **Source Canvas (`#source`)**:
  Captures pointer events to define normalized bounding boxes:
  ```javascript
  rect[mode] = {
    x: Math.min(p.x, drag.x),
    y: Math.min(p.y, drag.y),
    w: Math.max(0.01, Math.abs(p.x - drag.x)),
    h: Math.max(0.01, Math.abs(p.y - drag.y))
  };
  ```
- **Preview Canvas (`#preview`, 540×960)**:
  Renders at 60fps via `requestAnimationFrame`. Uses the `cover()` function to calculate aspect ratio scaling without distortion:
  ```javascript
  function cover(source, r, x, y, w, h) {
    const sw = r.w * sourceW, sh = r.h * sourceH;
    const ratio = Math.max(w / sw, h / sh);
    const cw = w / ratio, ch = h / ratio;
    pc.drawImage(source, r.x * sourceW + (sw - cw) / 2, r.y * sourceH + (sh - ch) / 2, cw, ch, x, y, w, h);
  }
  ```

---

### Phase 6: FFmpeg Complex Filtergraph Compilation

The backend converts the layout spec and normalized boxes into a single multi-stream FFmpeg filter string:

```python
def render_graph(name, rects, top=.32):
    spec = geometry(name, top)
    if spec['mode'] == 'blur':
        return (
            '[0:v]split=2[b][f];'
            '[b]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=8:2,scale=1080:1920[bg];'
            '[f]scale=1080:1920:force_original_aspect_ratio=decrease[fg];'
            '[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]'
        )
    layers = spec['layers']
    parts = []
    if len(layers) > 1:
        parts.append('[0:v]split=' + str(len(layers)) + ''.join(f'[s{i}]' for i in range(len(layers))))
    for i, layer in enumerate(layers):
        source = f'[s{i}]' if len(layers) > 1 else '[0:v]'
        crop = ''
        if layer['source'] != 'full':
            x, y, w, h = rects[0 if layer['source'] == 'cam' else 1]
            crop = f'crop={w}:{h}:{x}:{y},'
        w, h = layer['w'], layer['h']
        mask = ''
        if layer.get('circle'):
            mask = ",format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W,H)/2,2)),255,0)'"
        parts.append(source + crop + f'scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1' + mask + f'[p{i}]')
    
    first = layers[0]
    parts.append(f"[p0]pad=1080:1920:{first['x']}:{first['y']}:black[b0]")
    for i, layer in enumerate(layers[1:], 1):
        parts.append(f"[b{i-1}][p{i}]overlay={layer['x']}:{layer['y']}:shortest=1[b{i}]")
    parts.append(f'[b{len(layers)-1}]null[base]')
    return ';'.join(parts)
```

#### Final Hardware-Accelerated Export Command
```bash
ffmpeg -y -ss <start> -to <end> -i <source.mp4> \
  -filter_complex "[filter_graph];[base]ass='<subtitles.ass>'[outv]" \
  -map "[outv]" -map 0:a? \
  -c:v h264_nvenc -preset p4 -cq 21 -b:v 6500k -maxrate 9000k -bufsize 14000k \
  -c:a aac -b:a 192k -ar 48000 \
  -movflags +faststart <output.mp4>
```
*(If NVIDIA NVENC fails or is not available, the encoder automatically falls back to `-c:v libx264 -crf 20 -preset fast`).*

---

### Phase 7: Adaptive Feedback Learning Loop (`learn.py`)

Under each generated clip, users can provide a 1–5 star rating. This directly trains the scoring model:
- **Features Tracked**: `loud`, `shout`, `words`, `motion`, `flash`, `strobe`, `cuts`, `llm`.
- **Default Weights**:
  ```python
  DEFAULT = {
      'loud': 0.17, 'shout': 0.13, 'words': 0.20, 'motion': 0.17,
      'flash': 0.08, 'strobe': 0.04, 'cuts': 0.03, 'llm': 0.18
  }
  ```
- **Ridge Regression Fitting**:
  When feedback reaches $\ge 8$ ratings, ridge regression computes a weight vector $w$ minimizing error against user star targets, penalized by deviation from the default baseline:
  $$w = (X^T X + \lambda I)^{-1} (X^T y + \lambda w_{\text{default}})$$
  Blends in gradually ($\alpha = \min(0.85, \frac{N}{60})$) so the engine gracefully adapts to personal taste without overfitting.

---

### Phase 8: Saved Clip Re-Edit Workflow (`reedit.py`)

Users can change framing, layout, font styles, and subtitle texts on exported clips **without re-downloading or re-transcribing**:
1. Frontend sends updated settings to `/clips/reedit`.
2. Re-uses the cached source video and existing `*.words.json`.
3. Compiles a new ASS subtitle file and FFmpeg filtergraph.
4. Generates a new 1080×1920 MP4 in under 15 seconds.

---

## 5. REST API Specifications

### `POST /link`
Initializes a video import and triggers parallel preview frame extraction.
- **Request Body**: `{"url": "https://www.youtube.com/watch?v=..."}`
- **Response**: `{"job": "3f9a1b..."}`

### `GET /status/{job}`
Polls progress for downloading, transcription, or rendering jobs.
- **Response**:
  ```json
  {
    "status": "done",
    "stage": "Source ready",
    "progress": 1.0,
    "id": "3f9a1b",
    "duration": 5420.5,
    "w": 1920,
    "h": 1080,
    "preview_frames": [
      {"time": 813.0, "url": "/preview/3f9a1b/0", "kind": "storyboard"}
    ]
  }
  ```

### `POST /make`
One-click autonomous pipeline: downloads, detects moments, transcribes, and renders 9:16 clips.
- **Request Body**:
  ```json
  {
    "url": "https://youtube.com/watch?v=...",
    "count": 4,
    "mode": "fast",
    "accurate": "accurate",
    "min_len": 30,
    "max_len": 60,
    "layout": "stack",
    "top": 0.32,
    "cam": {"x": 0.0, "y": 0.0, "w": 0.24, "h": 0.32},
    "game": {"x": 0.24, "y": 0.0, "w": 0.76, "h": 1.0},
    "caption_style": "gta",
    "language": ""
  }
  ```
- **Response**: `{"job": "7c2e4d..."}`

### `POST /transcribe`
Transcribes a specific time interval.
- **Request Body**: `{"id": "source_id", "start": 10.0, "end": 45.0, "language": "hinglish"}`
- **Response**: `{"job": "9e1a2f..."}` (Returns `words` list with start/end/text when done).

### `POST /export`
Manually renders a selected time range with specific layout and caption parameters.
- **Request Body**:
  ```json
  {
    "id": "source_id",
    "layout": "stack",
    "top": 0.32,
    "cam": {"x": 0, "y": 0, "w": 0.25, "h": 0.3},
    "game": {"x": 0.25, "y": 0, "w": 0.75, "h": 1.0},
    "start": 10.0,
    "end": 40.0,
    "captions": true,
    "caption_style": "gta",
    "caption_y": 1450,
    "words": [{"start": 0.2, "end": 0.8, "text": "AMAZING"}]
  }
  ```
- **Response**: `{"job": "..."}` $\rightarrow$ Final payload: `{"file": "clip_id.mp4", "srt": "/text/clip_id/srt"}`.

### `POST /feedback`
Records user star rating and updates regression model.
- **Request Body**: `{"file": "clip_id.mp4", "rating": 5}`
- **Response**: `{"ok": true, "ratings": 12, "fitted": true, "insight": ["Shouting +0.06"]}`

---

## 6. Directory Structure & File Manifest

```
neonclipping/
├── app.py                  # Main FastAPI server and routing
├── standalone.py           # Background task runner & headless CLI orchestration
├── layouts.py              # 11 Native layout definitions & FFmpeg filter graph compiler
├── highlights.py           # Audio energy, scream FFT band & optical flow ranking
├── action.py               # Low-res Farneback optical flow, cuts, flash & strobe detection
├── excite.py               # Lexicon of viral exclamations (English, Hindi, Hinglish)
├── smart.py                # Ollama LLM client, sentence snapping, title & hook generator
├── captions.py             # faster-whisper ASR loader, word grouping & ASS/SRT generator
├── captions_yt.py          # Instant YouTube auto-captions json3 extractor
├── translit.py             # Rule-based Devanagari to Hinglish phonetic transliterator
├── learn.py                # User rating feedback loop & ridge regression re-weighting
├── previews.py             # Storyboard thumbnail extractor for instant UI preview
├── reedit.py               # Layout/subtitle editor for previously exported clips
├── index.html              # Frontend user interface
├── style.css               # Neon-styled dark theme UI styling
├── app.js                  # Frontend client state, pointer dragging & 60fps canvas loop
├── requirements.txt        # Python dependency manifest
├── cookies.txt             # (Optional) Exported YouTube cookies for anti-bot bypass
└── data/                   # Working directory for sources, clips, transcripts, and logs
    ├── *.mp4               # Downloaded source videos and finished exports
    ├── *.words.json        # Cached word-level transcripts
    ├── *.state.json        # Persistent job tracking states
    ├── feedback.jsonl      # User star ratings log
    └── weights.json        # Re-fitted adaptive feature weights
```

---

## 7. How to Deploy & Replicate in Another Project

### Step 1: System Dependencies
Ensure the following binaries are installed and on your system `PATH`:
- **Python 3.10 or 3.11** (Python 3.11 recommended for CUDA PyTorch wheels).
- **FFmpeg 6.0+** (Compiled with `--enable-libass` and `--enable-nvenc`).
- **NVIDIA GPU Drivers** (CUDA 12+ recommended for hardware acceleration).
- **Ollama** installed and running on port 11434 (`ollama run qwen3.5:4b`).

### Step 2: Python Virtual Environment & Packages
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

#### `requirements.txt`
```txt
fastapi>=0.110.0
uvicorn>=0.28.0
yt-dlp>=2024.03.10
opencv-python-headless>=4.9.0.80
faster-whisper>=1.0.0
ctranslate2>=4.0.0
onnxruntime>=1.17.0
Pillow>=10.2.0
numpy>=1.26.0
```

### Step 3: Run the Server
```bash
python app.py
```
Open **`http://127.0.0.1:8770`** in your browser.

---

## 8. Troubleshooting & Edge Cases

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **`Sign in to confirm you’re not a bot`** | YouTube IP bot detection challenge. | Export your browser's YouTube cookies into `cookies.txt` in the project root using a cookie export extension. |
| **`CUDA out of memory`** | `large-v3` model spills VRAM on GPUs $< 6\text{ GB}$. | Set `accurate` dropdown to `Fast` or `Accurate (turbo)`, or launch with `NEONCLIPPING_CPU=1`. |
| **Subtitles out of sync** | Video had variable frame rate (VFR). | The ingestion pipeline automatically normalizes videos using FFmpeg `-movflags +faststart` and fixed audio sample rates (`48000 Hz`). |
| **Ollama connection refused** | Ollama daemon is not running. | Run `ollama serve` in a terminal. If absent, the engine falls back gracefully to audio/motion heuristic scoring. |

---


