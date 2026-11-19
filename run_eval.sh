#!/usr/bin/env bash
# Run eval_libero.py with the LIBERO environment from ../vla-qwen (.env-libero), headless EGL.
# GPU=<id> selects the GPU (default 0).
cd "$(dirname "$0")"
VLA="$(cd ../vla-qwen && pwd)"
export PYTHONPATH="$VLA/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$VLA/third_party/libero_cfg"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${GPU:-0}" MUJOCO_EGL_DEVICE_ID="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
exec "$VLA/.env-libero/bin/python" -u eval_libero.py "$@" 2> >(grep -vE "robosuite WARNING|EGLError|egl_context|glCheckError|OpenGL/error|Exception ignored|Traceback \(most recent call last\):$|^\s+(File|\^|self\.|raise|EGL)" >&2)
