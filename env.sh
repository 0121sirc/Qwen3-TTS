#!/usr/bin/env bash
# Shared runtime environment for the Qwen3-TTS launchers:
#     webui.sh, openai_api_server.sh
#
# Source it, do not execute it (from the repo root):
#     source ./env.sh
#
# This file lives at the repo ROOT (not inside .conda_env/) so that it survives
# a fresh clone: .conda_env/ is gitignored because it holds the conda env itself,
# but this file is hand-written config the launchers cannot start without.
#
# Two concerns live here (each toggleable from the environment before sourcing):
#   1. PATH   -> the env's bin dir, so gradio/ffmpeg can find ffmpeg + ffprobe
#                (mp3/flac/opus responses of openai_tts_server.py need ffmpeg)
#   2. malloc -> glibc arena/mmap tuning; cheap insurance against the model load
#                (bf16 weights ~2.4GB + torch buffers, peak ~3.5GB) fragmenting
#                a 7.8GB machine
#
# No cuDNN/LD_LIBRARY_PATH block here: the Qwen3-TTS stack runs everything in
# torch (speech tokenizer included), so onnxruntime is CPU-only and the pip
# torch wheel carries its own CUDA libs via RPATH.
#
# Safe to source more than once.
[[ -n "${QWEN3_TTS_ENV_SOURCED:-}" ]] && return 0
QWEN3_TTS_ENV_SOURCED=1

# This file sits at the repo root, but the interpreter it puts on PATH lives in
# the (gitignored) .conda_env/, so anchor to ./env.sh -> ./.conda_env explicitly.
_QWEN3_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_QWEN3_ENV_DIR="$_QWEN3_ROOT/.conda_env"

# --- 1. PATH -----------------------------------------------------------------
export PATH="$_QWEN3_ENV_DIR/bin:$PATH"

# --- 2. glibc malloc ---------------------------------------------------------
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-131072}"
export MALLOC_MMAP_MAX_="${MALLOC_MMAP_MAX_:-65536}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-67108864}"

# --- 3. misc -----------------------------------------------------------------
# gradio/torch print a fork() warning otherwise.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset _QWEN3_ENV_DIR
unset _QWEN3_ROOT
