"""
LinguaFlow Backend v4 — Adaptive VAD + Noise Gate + Translation Cache + Context
=================================================================================

Architecture:
  Browser mic → ScriptProcessorNode (16kHz Int16 PCM) → Binary WebSocket
  → silero-vad (adaptive threshold) → noise gate → faster-whisper ASR
  → cache lookup → LLM streaming translation (with conversation context) → TTS

Key improvements over v3:
  - Adaptive VAD: tracks noise floor, adjusts threshold dynamically
  - Noise gate: attenuates low-energy frames before ASR to reduce false recognition
  - Pre-roll buffer: preserves speech onset that would otherwise be clipped
  - Translation cache: LRU-style dict, avoids re-translating repeated phrases
  - Conversation context: injects last 5 translations into system prompt for coherence
  - Language-specific translation rules for better accuracy

Usage:
    pip install fastapi uvicorn faster-whisper numpy httpx pydantic torch torchaudio
    python server.py
"""

import asyncio
import io
import json
import logging
import struct
import time
import wave
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# ---------------------------------------------------------------------------
# Logging — structured, with timestamps. Production should suppress transcript
# content for privacy; currently INFO-level for debugging.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("linguaflow")

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE    = 16000      # Target sample rate for all audio processing
VAD_FRAME_MS   = 32         # silero-vad frame duration (32ms = 512 samples @16kHz)
VAD_THRESHOLD  = 0.5        # Base VAD threshold — adaptive code adjusts dynamically

# ---- Adaptive VAD parameters ----
NOISE_FLOOR_ALPHA  = 0.05   # EMA smoothing factor for noise floor (slow adaptation)
NOISE_UPDATE_EVERY = 50     # Re-estimate noise floor every N silence frames
VAD_MIN_THRESHOLD  = 0.3    # Lower bound for adaptive threshold (quiet environments)
VAD_MAX_THRESHOLD  = 0.7    # Upper bound (noisy environments)
VAD_MARGIN         = 0.15   # Threshold = noise_floor + margin

# ---- VAD cutting parameters ----
# "微型分段"策略实现流式ASR效果：
#   说话中每~1s强制切段 → 立即ASR → 前端实时拼接显示
#   静音~256ms触发最终切段 → 完整转录 → 触发翻译
SILENCE_FRAMES_TO_STOP = 8     # ~256ms silence triggers utterance end + translation
MIN_SPEECH_FRAMES      = 10    # Minimum speech duration (320ms) — rejects clicks
MAX_SPEECH_FRAMES      = 30    # ~1s force-cut → streaming partial transcript
PRE_ROLL_FRAMES        = 5     # Pre-roll frames stored before speech onset (anti-clip)

# ---- ASR parameters ----
# Accuracy-optimized. small = best CPU-viable model for French + multilingual.
# If too slow on your machine, drop back to "base" or "tiny".
BEAM_SIZE = 2                  # Beam search width (higher = more accurate)
WHISPER_MODEL_SIZE = "small"   # small (500MB) >> base (140MB) > tiny (80MB)

# ---- Noise gate parameters ----
NOISE_GATE_DB  = -40.0         # dBFS threshold — frames below this are attenuated
NOISE_GATE_RATIO = 0.5         # Attenuation ratio (0.5 = -6dB reduction on quiet frames)

# ---- Translation cache ----
CACHE_MAX_SIZE = 200           # Max cached translations before evicting oldest
_cache: OrderedDict[str, str] = OrderedDict()  # key = (text, src, tgt), value = translation

# ---- Conversation context ----
CONTEXT_WINDOW = 5             # Number of previous exchanges sent as context

# ============================================================================
# Device / GPU detection
# ============================================================================

def detect_device() -> tuple[str, str]:
    """
    Auto-detect the best available compute device.
    Priority: CUDA > MPS (Apple Silicon) > CPU.
    MPS falls back to CPU because ctranslate2's MPS support is experimental and unstable.
    Returns (device_name, compute_type) for faster-whisper.
    """
    if torch.cuda.is_available():
        logger.info("GPU detected: CUDA — using float16")
        return "cuda", "float16"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple MPS detected — falling back to CPU (MPS unstable in ctranslate2)")
        return "cpu", "int8"
    logger.info("No GPU — using CPU with int8 quantization")
    return "cpu", "int8"

DEVICE, COMPUTE_TYPE = detect_device()

# ============================================================================
# Lazy-loaded models (loaded on first WebSocket connection)
# ============================================================================

_whisper_model: Optional[object] = None  # faster-whisper WhisperModel singleton
_vad_model:     Optional[object] = None  # silero-vad JIT model singleton

def load_whisper():
    """
    Singleton loader for faster-whisper. First call downloads the model from
    HuggingFace (~140MB for 'base'), subsequent calls return cached instance.
    Uses 4 CPU threads for parallel inference on multi-core machines.
    """
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper ({WHISPER_MODEL_SIZE}) on {DEVICE}...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=4,
        )
        logger.info("faster-whisper loaded.")
    return _whisper_model

def load_vad():
    """
    Singleton loader for silero-vad. Uses torch.hub to download/cache the model.
    The model is a JIT-traced script module — runs on CPU only (~2MB).
    Set torch num_threads to 1 to avoid contention with faster-whisper threads.
    """
    global _vad_model
    if _vad_model is None:
        torch.set_num_threads(1)
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _vad_model = model
        logger.info("silero-vad loaded.")
    return _vad_model

def get_vad_model():
    """Return the shared VAD model, loading it if necessary."""
    return load_vad()

# ============================================================================
# Data model: SpeechSegment — a single utterance queued for ASR
# ============================================================================

class SpeechSegment:
    """
    Represents one captured utterance.
    seg_id: monotonically increasing ID for frontend bubble matching.
    wav_data: WAV-encoded bytes (16kHz, mono, 16-bit PCM).
    timestamp: epoch time when the segment was captured.
    is_final: True if silence-cut (end of utterance, trigger translation).
              False if force-cut (mid-speech partial, display only).
    """
    def __init__(self, seg_id: int, wav_data: bytes, timestamp: float, is_final: bool = True):
        self.seg_id = seg_id
        self.wav_data = wav_data
        self.timestamp = timestamp
        self.is_final = is_final

# ============================================================================
# Per-WebSocket session — all state for one connected client
# ============================================================================

class TranscriptionSession:
    """
    Holds all mutable state for a single WebSocket connection.
    Each client gets one session. Sessions are isolated — no shared state
    except the singleton models (which are read-only during inference).
    """

    def __init__(self):
        # ---- VAD frame buffer ----
        self.speech_buffer:    list[np.ndarray] = []  # Accumulated float32 speech frames
        self.pre_roll_buffer:  list[np.ndarray] = []  # Frames stored BEFORE speech onset (anti-clip)
        self.silence_counter:  int = 0
        self.is_speaking:      bool = False
        self.total_speech_frames: int = 0

        # ---- Adaptive VAD state ----
        self._noise_floor:     float = 0.05    # EMA estimate of background noise level
        self._silence_since_update: int = 0     # Silence frames since last noise floor update
        self._adaptive_threshold: float = VAD_THRESHOLD  # Current effective threshold

        # ---- Pipeline (ASR worker + translation tasks) ----
        self.task_queue: asyncio.Queue[SpeechSegment] = asyncio.Queue()
        self._seg_id:    int = 0                     # Monotonic segment ID counter
        self._asr_worker: Optional[asyncio.Task] = None
        self._active_translations: set[asyncio.Task] = set()  # Running translation tasks
        self._utterance_parts: list[str] = []        # Accumulated partial texts for current utterance

        # ---- Conversation context (for coherent translation) ----
        self.context_buffer: list[str] = []           # Last N translated texts

        # ---- User configuration (set via WebSocket 'config' message) ----
        self.source_lang: str = "fr"
        self.target_lang: str = "zh"
        self.api_key:     str = ""
        self.api_base:    str = "https://api.openai.com/v1"
        self.model:        str = "gpt-4o-mini"
        self.whisper_pref: str = ""      # Frontend preference, applied at restart
        self.paused:       bool = False

    def next_seg_id(self) -> int:
        """Allocate the next segment ID. 1-based for human readability."""
        self._seg_id += 1
        return self._seg_id

    def reset_speech(self):
        """
        Reset VAD state after capturing a segment.
        Clears the speech buffer, silence counter, and pre-roll.
        Does NOT reset the noise floor estimate (that should persist).
        """
        self.speech_buffer.clear()
        self.pre_roll_buffer.clear()
        self.silence_counter = 0
        self.is_speaking = False
        self.total_speech_frames = 0

    def add_context(self, translation: str):
        """Add a finished translation to the context buffer (sliding window)."""
        if translation:
            self.context_buffer.append(translation)
            if len(self.context_buffer) > CONTEXT_WINDOW:
                self.context_buffer.pop(0)

    def get_speech_audio(self) -> Optional[bytes]:
        """
        Concatenate accumulated speech frames into a WAV byte buffer.
        Returns None if the segment is too short (rejects clicks/pops).
        Applies noise gate: attenuates low-energy frames to reduce
        background noise reaching the ASR model.
        """
        if not self.speech_buffer or self.total_speech_frames < MIN_SPEECH_FRAMES:
            return None

        audio = np.concatenate(self.speech_buffer)

        # ---- Noise gate: attenuate quiet frames ----
        rms = np.sqrt(np.mean(audio ** 2))  # Root-mean-square energy
        if rms > 1e-10:  # Avoid division by zero
            rms_db = 20.0 * np.log10(rms)  # Convert to dBFS
            if rms_db < NOISE_GATE_DB:
                # Attenuate: multiply by ratio (0.5 = halve amplitude = -6dB)
                audio = audio * 0.5

        # ---- Convert float32 [-1,1] → int16 [-32768,32767] ----
        audio_int16 = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

# ============================================================================
# Audio processing helpers
# ============================================================================

def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert raw 16-bit little-endian PCM bytes to float32 in [-1.0, 1.0].
    Uses struct.unpack for speed (one pass, native C).
    """
    samples = len(pcm_bytes) // 2
    arr = struct.unpack(f"<{samples}h", pcm_bytes)
    return np.array(arr, dtype=np.float32) / 32768.0

def get_vad_frames(audio: np.ndarray, frame_size: int) -> list[np.ndarray]:
    """
    Split a 1D audio array into overlapping frames of `frame_size` samples.
    No overlap between frames (stride = frame_size).
    Partial final frame is discarded.
    """
    frames = []
    for i in range(0, len(audio) - frame_size + 1, frame_size):
        frames.append(audio[i : i + frame_size])
    return frames

def _update_noise_floor(session: TranscriptionSession):
    """
    Update the adaptive noise floor estimate using recent silence frames.
    Uses EMA (exponential moving average): new_floor = α·current + (1-α)·old_floor.
    Only updates during extended silence to avoid speech frames polluting the estimate.
    """
    if not session.pre_roll_buffer:
        return
    # Average energy of recent silence frames
    recent_frames = session.pre_roll_buffer[-NOISE_UPDATE_EVERY:]
    if not recent_frames:
        return
    avg_energy = float(np.mean([np.sqrt(np.mean(f ** 2)) for f in recent_frames]))
    # EMA update
    session._noise_floor = (NOISE_FLOOR_ALPHA * avg_energy
                            + (1.0 - NOISE_FLOOR_ALPHA) * session._noise_floor)
    # Clamp threshold to safe range
    session._adaptive_threshold = max(VAD_MIN_THRESHOLD,
                                       min(VAD_MAX_THRESHOLD,
                                           session._noise_floor * 8.0 + VAD_MARGIN))

# ============================================================================
# Translation cache — simple LRU-style dict (OrderedDict)
# ============================================================================

def _cache_key(text: str, src: str, tgt: str) -> str:
    """Produce a cache key. Normalize whitespace to catch repeated phrases."""
    return f"{src}→{tgt}:{text.strip().lower()}"

def _cache_get(text: str, src: str, tgt: str) -> Optional[str]:
    """Look up a translation. Returns cached result or None."""
    key = _cache_key(text, src, tgt)
    if key in _cache:
        # Move to end (most recently used)
        _cache.move_to_end(key)
        return _cache[key]
    return None

def _cache_set(text: str, src: str, tgt: str, translation: str):
    """Store a translation. Evicts oldest if cache is full."""
    key = _cache_key(text, src, tgt)
    if key in _cache:
        _cache.move_to_end(key)
    _cache[key] = translation
    if len(_cache) > CACHE_MAX_SIZE:
        _cache.popitem(last=False)  # Remove oldest (LRU eviction)

# ============================================================================
# Streaming translation (SSE) with cache + context
# ============================================================================

# Human-readable language names for system prompt
LANG_NAMES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "ru": "Russian", "ar": "Arabic", "hi": "Hindi", "th": "Thai",
    "vi": "Vietnamese", "it": "Italian", "nl": "Dutch", "auto": "the detected language",
}

# Language-pair-specific translation hints
def _get_lang_hints(src: str, tgt: str) -> str:
    """Return language-specific translation rules based on the language pair."""
    hints = []
    pair = f"{src}→{tgt}"

    if src == "fr":
        hints.append("French negation (ne...pas) often condenses in spoken form — interpret accordingly.")
    if src in ("ja", "ko"):
        hints.append("Subject pronouns are often omitted — infer from context.")
    if tgt == "zh":
        hints.append("Use natural spoken Chinese, not written-style. Prefer short sentences.")
    if "en" in pair:
        hints.append("Distinguish formal/informal register based on the original tone.")
    if src == "auto":
        hints.append("The source language is auto-detected — verify that the translation direction feels right.")

    return " ".join(hints) if hints else ""

async def translate_stream(
    text: str,
    source_lang: str,
    target_lang: str,
    api_key: str,
    api_base: str,
    model: str,
    context: list[str],
    on_token,
) -> str:
    """
    Stream translation from an OpenAI-compatible API.
    - Checks cache first (0-latency hit).
    - Injects conversation context for coherence.
    - Parses SSE (Server-Sent Events) line by line.
    - Calls on_token(partial_text) for each accumulated chunk.
    Returns the complete translation, or empty string on failure.
    """
    import httpx

    # ---- Cache lookup ----
    cached = _cache_get(text, source_lang, target_lang)
    if cached:
        logger.info(f"Cache hit: \"{text[:50]}...\" → \"{cached[:50]}...\"")
        # Simulate streaming for cached result (one big chunk)
        await on_token(cached)
        return cached

    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    lang_hints = _get_lang_hints(source_lang, target_lang)

    # ---- Build system prompt with conversation context ----
    context_block = ""
    if context:
        context_block = "Previous translations (for context only, do NOT repeat):\n"
        for i, prev in enumerate(context, 1):
            context_block += f"  [{i}] {prev}\n"
        context_block += "Use the above context to maintain pronoun references and consistency.\n"

    system_prompt = (
        f"You are a professional simultaneous interpreter. "
        f"Translate spoken text from {src_name} to {tgt_name}. "
        f"Output ONLY the translation — no explanations, no notes. "
        f"Preserve tone, intent, nuance. Use natural conversational {tgt_name}. "
        f"Handle idioms and colloquialisms. "
        f"If the input is fragmented, give your best interpretation. "
        f"{lang_hints} "
        f"{context_block}"
    )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": 600,
        "stream": True,
    }

    collected: list[str] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(f"Translation API error: {resp.status_code} {body[:200]}")
                return ""

            # Parse SSE stream: each "data: {...}" line is one JSON chunk
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":  # End-of-stream sentinel
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected.append(content)
                            await on_token("".join(collected))
                    except (json.JSONDecodeError, KeyError):
                        continue

    result = "".join(collected)
    if result:
        _cache_set(text, source_lang, target_lang, result)
    return result

# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(title="LinguaFlow Backend", version="4.0.0")


@app.get("/health")
async def health():
    """Health check endpoint. Returns device info + whisper model size."""
    return {
        "status": "ok",
        "service": "linguaflow",
        "device": DEVICE,
        "whisper": WHISPER_MODEL_SIZE,
        "cache_size": len(_cache),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Main WebSocket handler — one per connected browser tab.
    Lifecycle: accept → load models → start ASR worker → message loop → cleanup.
    """
    await ws.accept()
    session = TranscriptionSession()
    frame_size = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)  # = 512 samples

    # ---- Load models (blocking, but once per process) ----
    try:
        load_whisper()
        load_vad()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        await ws.send_json({"type": "error", "message": f"Model loading failed: {e}"})
        await ws.close()
        return

    logger.info("WebSocket connected")

    # ---- Convenience: non-failing JSON send ----
    async def send_json(msg: dict):
        try:
            await ws.send_json(msg)
        except Exception:
            pass  # Client may have disconnected mid-send

    # ------------------------------------------------------------------
    # Translation task — fire-and-forget background coroutine
    # ------------------------------------------------------------------
    async def run_translation(seg_id: int, text: str, ts: int):
        """
        Launch an independent translation for a single segment.
        - Checks cache
        - Calls LLM with streaming
        - Pushes partial/final results to frontend via send_json
        - On completion, adds to conversation context
        This runs concurrently with ASR — does NOT block the ASR worker.
        """
        if not session.api_key:
            await send_json({
                "type": "translation_end", "seg_id": seg_id,
                "text": "", "timestamp": ts, "note": "API key not configured",
            })
            return

        # Callback invoked by translate_stream on each new token
        async def on_token(partial: str):
            await send_json({
                "type": "translation_partial", "seg_id": seg_id,
                "text": partial, "timestamp": ts,
            })

        try:
            full = await translate_stream(
                text,
                session.source_lang,
                session.target_lang,
                session.api_key,
                session.api_base,
                session.model,
                list(session.context_buffer),  # Snapshot of current context
                on_token,
            )
            if full:
                logger.info(f"[{seg_id}] Translation: {full}")
                session.add_context(full)  # Feed back into context for next turn
        except asyncio.CancelledError:
            return  # User stopped recording; let it die silently
        except Exception as e:
            logger.error(f"[{seg_id}] Translation error: {e}")
        finally:
            # Always send a closing message so frontend stops the streaming cursor
            await send_json({
                "type": "translation_end", "seg_id": seg_id,
                "text": full if 'full' in dir() else "", "timestamp": ts,
            })
            session._active_translations.discard(asyncio.current_task())

    # ------------------------------------------------------------------
    # ASR worker — serial, non-blocking, spawns translation tasks
    # ------------------------------------------------------------------
    async def asr_worker():
        """
        Main ASR loop. Runs as a single background task consuming from
        session.task_queue. Processing is serial within this worker —
        but translation tasks run independently, so a slow translation
        doesn't block the next segment's ASR.
        """
        while True:
            segment: SpeechSegment = await session.task_queue.get()
            seg_id = segment.seg_id
            loop = asyncio.get_event_loop()
            model = _whisper_model

            try:
                await send_json({"type": "status", "message": "transcribing"})

                # ---- faster-whisper transcription (CPU-bound → run in executor) ----
                audio_file = io.BytesIO(segment.wav_data)
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: model.transcribe(
                        audio_file,
                        beam_size=BEAM_SIZE,
                        language=session.source_lang if session.source_lang != "auto" else None
                    ),
                )

                # Join all transcribed segments into one string
                full_text = " ".join(seg.text.strip() for seg in segments)
                if not full_text:
                    await send_json({"type": "status", "message": "listening"})
                    continue

                # When auto-detecting, use Whisper's result as the effective source language.
                # This gives the translation prompt a concrete language name instead of
                # "the detected language", improving translation quality.
                effective_src = session.source_lang
                if effective_src == "auto" and hasattr(info, 'language_probability'):
                    if info.language_probability > 0.5:
                        effective_src = info.language  # e.g. "fr", "zh"
                        # Persist as sticky detection for subsequent segments
                        session.source_lang = effective_src
                        logger.info(f"[{seg_id}] Auto-detect locked to {effective_src} "
                                    f"(confidence={info.language_probability:.2f})")

                logger.info(f"[{seg_id}] ASR [{info.language}] {'FINAL' if segment.is_final else 'PARTIAL'}: {full_text}")
                ts = int(time.time() * 1000)

                if segment.is_final:
                    # ---- Silence-cut: concatenate all partials + this segment ----
                    session._utterance_parts.append(full_text)
                    complete_text = " ".join(session._utterance_parts)
                    session._utterance_parts.clear()
                    logger.info(f"[{seg_id}] Final transcript (from {len(session._utterance_parts)+1} parts): {complete_text[:80]}...")

                    await send_json({
                        "type": "transcript", "seg_id": seg_id,
                        "text": complete_text,
                        "language": info.language,
                        "detected_lang": effective_src,
                        "timestamp": ts,
                    })
                    await send_json({
                        "type": "translation_start", "seg_id": seg_id, "timestamp": ts,
                    })
                    t_task = asyncio.create_task(run_translation(seg_id, complete_text, ts))
                    session._active_translations.add(t_task)
                else:
                    # ---- Force-cut: partial → accumulate for final concatenation ----
                    session._utterance_parts.append(full_text)
                    await send_json({
                        "type": "transcript_partial", "seg_id": seg_id,
                        "text": full_text, "language": info.language,
                        "timestamp": ts,
                    })
            except Exception as e:
                logger.error(f"[{seg_id}] ASR error: {e}")
                await send_json({"type": "error", "message": str(e)})
            finally:
                await send_json({"type": "status", "message": "listening"})

    # Start the ASR worker
    session._asr_worker = asyncio.create_task(asr_worker())

    # ------------------------------------------------------------------
    # Message loop — dispatches incoming WebSocket frames
    # ------------------------------------------------------------------
    try:
        while True:
            raw = await ws.receive()

            # ---- Binary frame = raw Int16 PCM from browser microphone ----
            if "bytes" in raw:
                if session.paused:
                    continue
                audio_float = pcm_to_float32(raw["bytes"])
                await _process_vad(audio_float, session, send_json, frame_size)
                continue

            # ---- Text frame = JSON control message ----
            if "text" not in raw:
                continue

            msg = json.loads(raw["text"])
            msg_type = msg.get("type", "")

            if msg_type == "config":
                # Client sends config on connect + after settings changes
                session.source_lang = msg.get("sourceLang", session.source_lang)
                session.target_lang = msg.get("targetLang", session.target_lang)
                session.api_key     = msg.get("apiKey", session.api_key)
                session.api_base    = msg.get("apiBase", session.api_base)
                session.model       = msg.get("model", session.model)
                session.whisper_pref = msg.get("whisperSize", "")
                if session.whisper_pref and session.whisper_pref != WHISPER_MODEL_SIZE:
                    logger.info(f"Whisper preference: {session.whisper_pref} (currently {WHISPER_MODEL_SIZE}). "
                                "Restart server to apply.")
                logger.info(f"Config: {session.source_lang}→{session.target_lang}, model={session.model}")
                await send_json({"type": "status", "message": "listening"})

            elif msg_type == "audio":
                # Legacy Base64 audio frame (backward compat)
                if session.paused:
                    continue
                pcm_b64 = msg.get("data", "")
                if not pcm_b64:
                    continue
                try:
                    import base64
                    pcm_bytes = base64.b64decode(pcm_b64)
                except Exception:
                    continue
                audio_float = pcm_to_float32(pcm_bytes)
                await _process_vad(audio_float, session, send_json, frame_size)

            elif msg_type == "stop":
                # User clicked stop: flush buffer, cancel ASR worker, drain queue
                if session.is_speaking:
                    _enqueue_speech(session)
                else:
                    session.reset_speech()
                if session._asr_worker and not session._asr_worker.done():
                    session._asr_worker.cancel()
                    try:
                        await session._asr_worker
                    except asyncio.CancelledError:
                        pass
                # Drain stale queue items
                while not session.task_queue.empty():
                    try:
                        session.task_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                # Restart ASR worker (fresh state for next recording)
                session._asr_worker = asyncio.create_task(asr_worker())
                await send_json({"type": "status", "message": "listening"})

            elif msg_type == "pause":
                session.paused = True
                if session.is_speaking:
                    _enqueue_speech(session)

            elif msg_type == "resume":
                session.paused = False

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup: cancel all running tasks
        if session._asr_worker:
            session._asr_worker.cancel()
        for t in list(session._active_translations):
            t.cancel()
        try:
            await asyncio.gather(
                session._asr_worker,
                *session._active_translations,
                return_exceptions=True,
            )
        except Exception:
            pass

# ============================================================================
# VAD processing — the real-time speech detection loop
# ============================================================================

async def _process_vad(
    audio_float: np.ndarray,
    session: TranscriptionSession,
    send_json,
    frame_size: int,
):
    """
    Process one chunk of incoming audio through the VAD state machine.
    Called for every binary/text audio frame from the browser (~11 times/sec).

    State machine:
      NOT SPEAKING → confidence > threshold → SPEAKING (start accumulating)
      SPEAKING     → silence > N frames       → end segment (enqueue for ASR)
                    → frames > MAX            → force-cut (enqueue for ASR)

    Adaptive threshold: tracks noise floor during silence, adjusts upward
    in noisy environments and downward in quiet ones.
    """
    vad_model = get_vad_model()
    frames = get_vad_frames(audio_float, frame_size)

    with torch.no_grad():  # Disable gradient tracking (inference only)
        for frame in frames:
            try:
                tensor = torch.from_numpy(frame).unsqueeze(0)  # Shape: [1, 512]
                speech_prob = vad_model(tensor, SAMPLE_RATE).item()  # Scalar 0-1
            except Exception as e:
                logger.error(f"VAD error: {e}")
                continue

            active_threshold = session._adaptive_threshold

            if speech_prob > active_threshold:
                # ---- Speech frame detected ----
                if not session.is_speaking:
                    # Speech onset: prepend pre-roll buffer (catches clipped onset)
                    session.speech_buffer.extend(session.pre_roll_buffer)
                    session.pre_roll_buffer.clear()
                    session.is_speaking = True
                    await send_json({"type": "status", "message": "speaking"})

                session.speech_buffer.append(frame)
                session.silence_counter = 0
                session.total_speech_frames += 1

                # Force-cut on long segments (max ~5s)
                if session.total_speech_frames >= MAX_SPEECH_FRAMES:
                    _enqueue_speech(session, is_final=False)  # Mid-speech force-cut

            else:
                # ---- Silence / non-speech frame ----
                if session.is_speaking:
                    # Currently speaking: accumulate silence counter
                    session.silence_counter += 1
                    session.speech_buffer.append(frame)

                    if session.silence_counter >= SILENCE_FRAMES_TO_STOP:
                        # Enough silence → end of utterance (final)
                        logger.info(
                            f"Segment end FINAL ({session.total_speech_frames} frames, "
                            f"~{session.total_speech_frames * VAD_FRAME_MS}ms, "
                            f"threshold={active_threshold:.3f})"
                        )
                        _enqueue_speech(session, is_final=True)
                else:
                    # Not speaking: maintain pre-roll buffer (anti-clip buffer)
                    # This preserves audio BEFORE speech onset so the first
                    # few phonemes aren't lost
                    session.pre_roll_buffer.append(frame)
                    if len(session.pre_roll_buffer) > PRE_ROLL_FRAMES:
                        session.pre_roll_buffer.pop(0)

                    # Periodically update noise floor estimate
                    session._silence_since_update += 1
                    if session._silence_since_update >= NOISE_UPDATE_EVERY:
                        _update_noise_floor(session)
                        session._silence_since_update = 0

def _enqueue_speech(session: TranscriptionSession, is_final: bool = True):
    """
    Take the current speech buffer, convert to WAV bytes (with noise gate),
    assign a segment ID, and push to the ASR task queue.

    is_final=True: silence-cut → send transcript + trigger translation
    is_final=False: force-cut → send transcript_partial (frontend appends to current bubble)

    Edge case handling:
    - Queue full: drop the oldest unprocessed segment to make room.
      This can happen if the user speaks much faster than ASR can process.
    """
    wav_data = session.get_speech_audio()
    session.reset_speech()
    if wav_data is not None:
        seg_id = session.next_seg_id()
        try:
            session.task_queue.put_nowait(SpeechSegment(seg_id, wav_data, time.time(), is_final))
        except asyncio.QueueFull:
            logger.warning("Task queue full — dropping oldest segment to make room")
            try:
                session.task_queue.get_nowait()  # Evict oldest
                session.task_queue.put_nowait(SpeechSegment(seg_id, wav_data, time.time(), is_final))
            except Exception:
                pass  # If even this fails, drop the segment (rare edge case)

# ============================================================================
# Entry point
# ============================================================================

def main():
    import uvicorn
    logger.info(
        f"LinguaFlow v4 on http://0.0.0.0:8765 "
        f"(device={DEVICE}, whisper={WHISPER_MODEL_SIZE}, beam={BEAM_SIZE}, "
        f"adaptive_vad=on, noise_gate=on, cache=on)"
    )
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

if __name__ == "__main__":
    main()
