#!/usr/bin/env python3
"""OpenAI-compatible TTS server for Qwen3-TTS.

Port of CosyVoice/openai_tts_server.py so any OpenAI TTS client (the ``openai``
SDK, an s2s pipeline, ...) can use this repo's ``Qwen3-TTS-12Hz-0.6B-Base``
weights and the preset voices stored under ``voices/`` + ``voices-ext/`` (merged
as a union).

Endpoints:
  GET  /v1/health        -> readiness probe
  GET  /v1/models        -> {"data": [{"id": ...}]}
  GET  /v1/audio/voices  -> {"voices": [...], "default": ...,
                             "details": {name: {"desc": ..., "kind": ...}}}
                             # display only; the prompt text is server-side
                             # input, never exposed
  GET  /v1/voices        -> identical body (alias; see the route below)
  POST /v1/audio/speech  -> pcm (chunked) | wav | mp3 | flac | opus

Request body (OpenAI shape + ``seed`` / ``params`` extensions)::

    {
      "model": "Qwen3-TTS-12Hz-0.6B-Base",   # optional, ignored
      "input": "要合成的文本",
      "voice": "default" | "bfy" | ...,
      "response_format": "pcm" | "wav" | "mp3" | "flac" | "opus",
      "speed": 1.0,
      "seed": 0,
      "instructions": "...",                  # custom_voice/voice_design checkpoints
      "params": {"language": "Chinese", "instruction": "..."}
    }

A voice is one directory under any voice root (only when the loaded checkpoint
is a *clone* checkpoint, i.e. ``tts_model_type == "base"``). Roots are scanned
in order and merged as a union (``voices/`` wins over ``voices-ext/`` on a name
collision)::

    voices/bfy/prompt.wav        # reference audio, 3-10s recommended, any sr
    voices/bfy/meta.json         # {"name": ..., "prompt_text": "...", "desc": "..."}
    voices-ext/mine/prompt.wav   # same layout, second root
    voices-ext/mine/meta.json

Backend selection per request:
  * ``tts_model_type == "base"``        -> clone voices/ directories
      - meta.prompt_text present -> ICL mode (best quality)
      - meta.prompt_text missing -> x-vector only mode (voice identity only)
  * ``tts_model_type == "custom_voice"`` -> preset speakers from the checkpoint
      (``/v1/audio/voices`` lists them; directories are not used)
  * ``tts_model_type == "voice_design"`` -> instruction-driven, voice is ignored

Differences from the CosyVoice server:
  * Qwen3-TTS cannot stream: the whole utterance is generated in one call, so
    ``pcm`` is chunked out *after* synthesis (time-to-first-byte == full RTF).
  * ``speed`` is not a model knob -- it is applied post-hoc with a phase vocoder
    (librosa), and a failure there logs a warning and returns the audio as-is.
  * There is no ``<|endofprompt|>`` marker and no text frontend flag; both are
    accepted for request compatibility and ignored.
  * ``params.language`` (default: Auto) maps to the model's language token.

The 6GB GPU cannot hold this server and the gradio webui at the same time --
openai_api_server.sh / webui.sh refuse to start while the other one is alive.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("qwen3tts.openai")

DEFAULT_MODEL_NAME = "Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_VOICE = "default"
# The model has no speed knob, so a non-default speed is a phase-vocoder stretch.
# webui.py in CosyVoice clamps to [0.5, 2.0]; keep the same envelope.
SPEED_MIN, SPEED_MAX = 0.5, 2.0
MIN_PROMPT_S = 1.0               # below this the speaker embedding is meaningless
SHORT_PROMPT_WARN_S = 3.0        # README recommends >= 3s for voice cloning
LONG_PROMPT_WARN_S = 30.0        # long refs inflate the prompt context, no hard assert
# 8192 output tokens at 12Hz ~ 680s of speech; long input is bounded so a runaway
# request cannot blow the KV cache out of the 6GB card.
SHORT_TEXT_WARN_S = 300
MAX_INPUT_CHARS = 800
# Voice roots: scanned in order and merged as a union; an earlier root wins when
# two roots define the same voice name (so ./voices overrides ./voices-ext).
DEFAULT_VOICE_DIRS = (HERE / "voices", HERE / "voices-ext")
NATIVE_FORMATS = ("pcm", "wav")
FFMPEG_FORMATS = ("mp3", "flac", "opus")
MEDIA_TYPES = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "opus": "audio/ogg",
}
# How many pre-computed voice prompts to keep resident (they hold a couple of
# small GPU tensors each -- ref tokens + speaker embedding -- so this is cheap
# and saves re-running the speech tokenizer on every request).
PROMPT_CACHE_MAX = 8

app = FastAPI(title="Qwen3-TTS OpenAI-compatible TTS")

_model = None
_mode: str | None = None                # "base" | "custom_voice" | "voice_design"
_sample_rate = 24000
_dtype = "bfloat16"
_device = "cuda:0"
_attn = "sdpa"
_load_lock = Lock()
# One synthesis at a time: the model already fills most of the 6GB card.
_synth_lock = Lock()
_SYNTH_WAIT_S = 180                     # RTF ~2 on this card => long text needs room
_voices: dict[str, dict] = {}
_voice_dirs: list[Path] = list(DEFAULT_VOICE_DIRS)
_model_dir = HERE / "pretrained_models" / DEFAULT_MODEL_NAME
_speaker_map: dict[str, str] = {}       # lowercased name -> name reported by the ckpt
_languages: set[str] = set()            # lowercased; empty = do not validate
_prompt_cache: dict[str, list] = {}


# --------------------------------------------------------------------------- model
def _ensure_loaded():
    global _model, _mode, _sample_rate, _speaker_map, _languages
    if _model is None:
        with _load_lock:
            if _model is None:
                if not _model_dir.is_dir():
                    raise HTTPException(status_code=500,
                                        detail=f"model dir not found: {_model_dir}")
                logger.info("loading Qwen3-TTS model from %s (dtype=%s, attn=%s, device=%s)",
                            _model_dir, _dtype, _attn, _device)
                _model = Qwen3TTSModel.from_pretrained(
                    str(_model_dir),
                    device_map=_device,
                    dtype=getattr(torch, _dtype),
                    attn_implementation=None if _attn in ("auto", "none") else _attn,
                )
                # "base" (voice clone) / "custom_voice" (preset speakers) /
                # "voice_design" (instruction only); decides what a voice means.
                _mode = str(getattr(_model.model, "tts_model_type", None)
                            or _model.model.config.tts_model_type)
                try:
                    _speaker_map = {str(s).lower(): str(s)
                                    for s in (_model.get_supported_speakers() or [])}
                except Exception as exc:  # no speaker table on this checkpoint
                    logger.info("no speaker table (%s)", exc)
                    _speaker_map = {}
                try:
                    _languages = {str(s).lower() for s in (_model.get_supported_languages() or [])}
                except Exception:
                    _languages = set()
                logger.info("Qwen3-TTS model loaded: mode=%s, %d speaker(s), %d language(s)",
                            _mode, len(_speaker_map), len(_languages))
    return _model


# --------------------------------------------------------------------------- voices
def _scan_voices() -> dict[str, dict]:
    """Merge every voice root: <root>/<name>/{prompt.*, meta.json} -> {name: info}.

    The roots form a union; when two roots define the same name the earlier root
    wins and the shadowed one is logged.
    """
    voices: dict[str, dict] = {}
    roots = [d for d in _voice_dirs if d.is_dir()]
    for missing in (d for d in _voice_dirs if not d.is_dir()):
        logger.info("voices root %s does not exist (skipping)", missing)
    if not roots:
        logger.warning("no voice root exists: %s", ", ".join(map(str, _voice_dirs)))
        return voices
    for root in roots:
        for voice_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if voice_dir.name in voices:
                logger.warning("duplicate voice %s: keeping %s, ignoring %s",
                               voice_dir.name, voices[voice_dir.name]["path"].parent, voice_dir)
                continue
            prompts = sorted(voice_dir.glob("prompt.*"))
            if not prompts:
                logger.warning("skipping voice %s: no prompt.* file", voice_dir)
                continue
            meta = {}
            meta_file = voice_dir / "meta.json"
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    logger.warning("voice %s: ignoring broken meta.json (%s)", voice_dir, exc)
            voice = {
                "kind": "clone",
                "path": prompts[0],
                "prompt_text": str(meta.get("prompt_text") or "").strip(),
                "desc": str(meta.get("desc") or "").strip(),
            }
            # Cached prompt key includes mtime so editing prompt.wav/meta.json
            # invalidates the cached VoiceClonePromptItem without a restart.
            voice["key"] = f"{voice['path']}|{voice['prompt_text']}|{prompts[0].stat().st_mtime_ns}"
            voices[voice_dir.name] = voice
    if voices:
        logger.info("loaded %d voices from %d root(s): %s",
                    len(voices), len(roots), ", ".join(voices))
    else:
        logger.warning("no voices found under %s", ", ".join(map(str, roots)))
    return voices


def _clone_voices() -> dict[str, dict]:
    """Current voice directories (only meaningful for a clone checkpoint)."""
    if not _voices:
        _voices.update(_scan_voices())
    return _voices


def _default_voice_name() -> str:
    if _mode == "custom_voice":
        # No directories to fall back on: use the checkpoint's first speaker so a
        # client that omits `voice` still gets audio instead of a 400.
        return sorted(_speaker_map.values())[0] if _speaker_map else DEFAULT_VOICE
    names = sorted(_clone_voices())
    if DEFAULT_VOICE in names:
        return DEFAULT_VOICE
    return names[0] if names else DEFAULT_VOICE


def _resolve_voice(name: str) -> dict:
    """Map a `voice` to something the loaded checkpoint can actually use."""
    key = (name or "").strip()
    if not key or key.lower() == "default":
        # "default" is a directory name in clone mode and an alias for "first
        # available" in preset mode (this checkpoint has no such directory).
        key = _default_voice_name()

    if _mode == "custom_voice":
        # Preset speakers only -- this checkpoint cannot clone from audio.
        raw = _speaker_map.get(key.lower())
        if raw is None:
            available = ", ".join(sorted(_speaker_map.values())) or "<none>"
            raise HTTPException(status_code=400,
                                detail=f"unknown voice {key!r}; {_model_dir.name} only has "
                                       f"preset speakers: {available} (reference-audio "
                                       f"voices need a clone checkpoint, e.g. "
                                       f"--model_dir {DEFAULT_MODEL_NAME})")
        return {"kind": "preset", "path": None, "speaker": raw, "prompt_text": "",
                "desc": "preset speaker", "key": raw}

    if _mode == "voice_design":
        # Style is driven by `instructions`; the voice name has no meaning here.
        return {"kind": "design", "path": None, "prompt_text": "", "desc": "",
                "key": key}

    voices = _clone_voices()
    voice = voices.get(key)
    if voice is None:
        available = ", ".join(sorted(voices)) or "<none>"
        raise HTTPException(status_code=400,
                            detail=f"unknown voice {key!r}; available: {available}")
    return voice


def _validate_prompt(voice_name: str, voice: dict) -> None:
    import torchaudio
    path = voice["path"]
    try:
        info = torchaudio.info(str(path))
    except Exception as exc:  # unreadable/corrupt audio
        raise HTTPException(status_code=400,
                            detail=f"voice {voice_name!r}: cannot read {path.name}: {exc}")
    if info.sample_rate <= 0 or info.num_frames <= 0:
        raise HTTPException(status_code=400,
                            detail=f"voice {voice_name!r}: {path.name} has no audio")
    seconds = info.num_frames / info.sample_rate
    if seconds < MIN_PROMPT_S:
        raise HTTPException(status_code=400,
                            detail=f"voice {voice_name!r}: prompt audio is {seconds:.2f}s, "
                                   f"shorter than {MIN_PROMPT_S}s (need enough speech to "
                                   f"estimate a speaker embedding); use 3-10s")
    if seconds < SHORT_PROMPT_WARN_S:
        logger.warning("voice %s: prompt is %.1fs; Qwen3-TTS recommends >= %ds "
                       "(clone similarity drops on short references)",
                       voice_name, seconds, SHORT_PROMPT_WARN_S)
    elif seconds > LONG_PROMPT_WARN_S:
        # No hard limit like CosyVoice's 30s assert, but the reference is encoded
        # into the prompt context, so very long refs only cost memory and latency.
        logger.warning("voice %s: prompt is %.1fs; long references are expensive and "
                       "often unhelpful -- 3-10s of clean single-speaker speech works best",
                       voice_name, seconds)


def _validate_text(text: str) -> None:
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"input is {len(text)} chars, over the {MAX_INPUT_CHARS} "
                                   f"char limit (long text risks exhausting the KV cache "
                                   f"on a 6GB card); split it into shorter requests")
    if len(text) > SHORT_TEXT_WARN_S:
        logger.warning("long input (%d chars); expect several minutes at RTF ~2", len(text))


def _validate_language(language: str | None) -> str | None:
    if not language or not _languages:
        return None
    if language.lower() == "auto" or language.lower() in _languages:
        return language
    raise HTTPException(status_code=400,
                        detail=f"unsupported language {language!r}; available: "
                               f"{', '.join(sorted(_languages))} (or 'Auto')")


# ---------------------------------------------------------------------- synthesis
def _seed_all(seed: int) -> None:
    """Qwen3-TTS has no seed parameter: seeding the RNGs makes repeats byte-stable."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _voice_prompt(voice: dict) -> list:
    """Reference prompt for a voice, built once and reused across requests.

    create_voice_clone_prompt() runs the speech tokenizer + speaker encoder on the
    reference audio; caching it is what keeps the warm RTF at ~2.0 instead of
    paying that cost on every synthesis.
    """
    tts = _ensure_loaded()
    key = voice["key"]
    cached = _prompt_cache.get(key)
    if cached is not None:
        # Refresh insertion order so eviction removes the least recently used one.
        _prompt_cache[key] = cached
        return cached
    items = tts.create_voice_clone_prompt(
        ref_audio=str(voice["path"]),
        ref_text=voice["prompt_text"] or None,
        x_vector_only_mode=not voice["prompt_text"],
    )
    if len(_prompt_cache) >= PROMPT_CACHE_MAX:
        _prompt_cache.pop(next(iter(_prompt_cache)), None)
    _prompt_cache[key] = items
    logger.info("voice %r: cached prompt (%s, %d item(s))", voice["path"].name,
                "ICL" if voice["prompt_text"] else "x-vector only", len(items))
    return items


def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-6 or audio.size == 0:
        return audio
    try:
        import librosa
        stretched = librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32), rate=speed)
        return np.asarray(stretched, dtype=np.float32)
    except Exception as exc:
        # Never fail a request over a cosmetic knob.
        logger.warning("speed %.2f post-processing failed (%s); returning unmodified audio",
                       speed, exc)
        return audio


def _synthesize(text: str, voice: dict, req: "SpeechRequest") -> tuple[np.ndarray, int]:
    """Generate one utterance. The caller owns _synth_lock."""
    tts = _ensure_loaded()
    extra = dict(req.params or {})
    language = _validate_language(str(extra.get("language") or "").strip() or None)
    instruction = (req.instructions or extra.get("instruction") or "").strip()

    speed = 1.0 if req.speed is None else float(req.speed)
    if not (SPEED_MIN <= speed <= SPEED_MAX):
        clamped = min(max(speed, SPEED_MIN), SPEED_MAX)
        logger.warning("speed %.3f outside [%s, %s]; clamping to %.1f",
                       speed, SPEED_MIN, SPEED_MAX, clamped)
        speed = clamped

    if req.seed is not None:
        _seed_all(int(req.seed))

    if voice["kind"] == "clone":
        if instruction:
            # generate_voice_clone has no instruct channel on any Qwen3-TTS size;
            # silently dropping it would hide a client that thinks it applied style.
            logger.warning("checkpoint %s has no instruction support; ignoring %r",
                           _model_dir.name, instruction)
        logger.info("voice clone synthesis (%d chars, voice=%s, language=%s)",
                    len(text), voice["path"].name, language or "Auto")
        wavs, sr = tts.generate_voice_clone(text=text, language=language,
                                            voice_clone_prompt=_voice_prompt(voice))
    elif voice["kind"] == "preset":
        logger.info("custom voice synthesis (%d chars, speaker=%s, language=%s)",
                    len(text), voice["speaker"], language or "Auto")
        # 0.6B-CustomVoice ignores `instruct` internally; larger checkpoints honor
        # it, so pass it through and let the model decide.
        wavs, sr = tts.generate_custom_voice(text=text, speaker=voice["speaker"],
                                             language=language, instruct=instruction or None)
    else:  # voice_design
        logger.info("voice design synthesis (%d chars, instruction=%r, language=%s)",
                    len(text), instruction, language or "Auto")
        wavs, sr = tts.generate_voice_design(text=text, instruct=instruction,
                                             language=language)

    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise HTTPException(status_code=500, detail="synthesis produced no audio")
    global _sample_rate
    if int(sr) != _sample_rate:
        logger.info("sample rate changed %d -> %d", _sample_rate, int(sr))
        _sample_rate = int(sr)
    return _apply_speed(audio, speed), _sample_rate


def _to_pcm16(audio: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if not samples.size:
        return b""
    return (samples * 32767.0).astype("<i2").tobytes()


def _pcm_chunks(pcm: bytes, seconds: float = 1.0) -> Iterator[bytes]:
    """Hand an already-synthesized buffer out in ~1s slices.

    Qwen3-TTS has no incremental decoder, so this is packaging rather than
    streaming: the client still waits a full RTF for the first byte.
    """
    step = max(2, int(_sample_rate * seconds) * 2)
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]


def _wav_bytes(pcm: bytes) -> bytes:
    import wave
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


def _ffmpeg_convert(pcm: bytes, fmt: str) -> bytes:
    """s16le mono -> mp3/flac/opus via the env's ffmpeg (conda install)."""
    cmd = ["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(_sample_rate),
           "-ac", "1", "-i", "-", "-f", fmt]
    if fmt == "opus":
        cmd += ["-b:a", "64k"]
    cmd += ["-"]
    try:
        proc = subprocess.run(cmd, input=pcm, capture_output=True, check=False)
    except FileNotFoundError:
        raise HTTPException(status_code=501,
                            detail="ffmpeg not found on PATH; use response_format 'pcm' or 'wav'")
    if proc.returncode != 0:
        raise HTTPException(status_code=500,
                            detail=f"ffmpeg failed for {fmt}: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout


# ------------------------------------------------------------------------- api
class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    response_format: str = "pcm"
    speed: float | None = None
    seed: int | None = None
    instructions: str | None = None
    params: dict | None = None


@app.get("/v1/health")
def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model_loaded": _model is not None,
        # The directory name, not a constant: switching --model_dir to another
        # checkpoint reports itself as such, so /v1/health tells you what is live.
        "model": _model_dir.name,
        "mode": _mode,
        "sample_rate": _sample_rate,
        "voices": sorted(_voices) or sorted(_scan_voices()),
    })


@app.get("/v1/models")
def models() -> JSONResponse:
    return JSONResponse({
        "object": "list",
        "data": [{
            "id": _model_dir.name,
            "object": "model",
            "owned_by": "Qwen",
        }],
    })


# OpenAI ships no voice-list endpoint at all; LocalAI documents
# /v1/audio/voices, while ElevenLabs-style and most community
# "OpenAI-compatible" clients guess /v1/voices. Serve one handler from both
# paths (identical body) so their probe does not 404.
@app.get("/v1/voices")
@app.get("/v1/audio/voices")
def voices() -> JSONResponse:
    if _mode == "custom_voice":
        # Directories cannot be cloned by this checkpoint; list what it can do.
        names = sorted(_speaker_map.values())
        return JSONResponse({
            "voices": names,
            "default": names[0] if names else "",
            "details": {
                name: {"desc": "preset speaker", "kind": "preset"}
                for name in names
            },
        })
    current = _scan_voices()
    _voices.clear()
    _voices.update(current)
    default = DEFAULT_VOICE if DEFAULT_VOICE in current else (next(iter(current), ""))
    return JSONResponse({
        "voices": sorted(current),
        "default": default,
        "details": {
            # display metadata only: prompt_text stays server-side (it is
            # synthesis input, never something a caller needs)
            name: {"desc": info["desc"], "kind": info["kind"]}
            for name, info in current.items()
        },
    })


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    text = (req.input or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty input")
    _validate_text(text)

    fmt = (req.response_format or "pcm").lower()
    if fmt not in NATIVE_FORMATS and fmt not in FFMPEG_FORMATS:
        raise HTTPException(status_code=400,
                            detail=f"response_format must be one of "
                                   f"{list(NATIVE_FORMATS + FFMPEG_FORMATS)}, got {req.response_format!r}")

    # Resolve the voice before taking the synthesis slot so a bad request never
    # makes a concurrent synthesis wait.
    _ensure_loaded()
    voice_name = req.voice or _default_voice_name()
    voice = _resolve_voice(voice_name)
    if voice["kind"] == "clone":
        _validate_prompt(voice_name, voice)

    logger.info("tts request: %d chars, voice=%s, format=%s, speed=%s, seed=%s",
                len(text), voice_name, fmt, req.speed, req.seed)

    # Generation is the only GPU work; release the slot before shipping bytes.
    if not _synth_lock.acquire(timeout=_SYNTH_WAIT_S):
        raise HTTPException(status_code=503,
                            detail=f"another synthesis is still running (waited {_SYNTH_WAIT_S}s)")
    try:
        audio, _sr = _synthesize(text, voice, req)
        pcm = _to_pcm16(audio)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        _synth_lock.release()

    if not pcm:
        raise HTTPException(status_code=500, detail="synthesis produced no audio")
    if fmt == "pcm":
        return StreamingResponse(_pcm_chunks(pcm), media_type=MEDIA_TYPES[fmt])

    payload = _wav_bytes(pcm) if fmt == "wav" else _ffmpeg_convert(pcm, fmt)
    return Response(content=payload, media_type=MEDIA_TYPES[fmt])


def main() -> None:
    global _voice_dirs, _model_dir, _dtype, _device, _attn
    parser = argparse.ArgumentParser(description="Qwen3-TTS OpenAI-compatible TTS server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--model_dir", default=str(HERE / "pretrained_models" / DEFAULT_MODEL_NAME))
    parser.add_argument("--voices_dirs", nargs="+", default=None,
                        help="voice roots merged as a union, earlier roots win on a name "
                             "collision (default: %s)"
                             % " ".join(str(p) for p in DEFAULT_VOICE_DIRS))
    parser.add_argument("--dtype", default=_dtype,
                        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
                        help="checkpoint dtype (bfloat16 is the only one that works: fp16 "
                             "diverges, fp32 needs 5.3GB)")
    parser.add_argument("--device", default=_device, help="device_map value, e.g. cuda:0")
    parser.add_argument("--attn", default=_attn,
                        choices=["sdpa", "eager", "flash_attention_2", "auto"],
                        help="attn_implementation (flash-attn cannot install on sm61)")
    args = parser.parse_args()

    _model_dir = Path(args.model_dir)
    _dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(args.dtype, args.dtype)
    _device = args.device
    _attn = args.attn
    if args.voices_dirs:
        _voice_dirs = [Path(p) for p in args.voices_dirs]
    _voices.update(_scan_voices())

    # Load before serving so /v1/health only answers once the model is usable.
    _ensure_loaded()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
