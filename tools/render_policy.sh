#!/usr/bin/env bash
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: tools/render_policy.sh checkpoints/model.pt"
  exit 2
fi
cd "$(dirname "$0")/../mjlab"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export HUD03_EASY_MIMIC=1
export HUD03_DAB_STRICT=${HUD03_DAB_STRICT:-1}
uv run play Mjlab-Tracking-Flat-LimX-HU-D03 \
  --agent trained \
  --checkpoint-file "../$1" \
  --motion-file "../motions/hud03_stand_dab_loop.npz" \
  --num-envs 1 \
  --video True \
  --video-length 360 \
  --video-height 480 \
  --video-width 640 \
  --no-terminations True \
  --log-root "../outputs/play"
