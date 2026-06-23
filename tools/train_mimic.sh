#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../mjlab"
export MUJOCO_GL=${MUJOCO_GL:-disable}
export HUD03_EASY_MIMIC=1
export HUD03_DAB_STRICT=${HUD03_DAB_STRICT:-1}
uv run train Mjlab-Tracking-Flat-LimX-HU-D03 \
  --env.commands.motion.motion-file "../motions/hud03_stand_dab_loop.npz" \
  --env.scene.num-envs "${HUD03_NUM_ENVS:-256}" \
  --agent.max-iterations "${HUD03_MAX_ITERATIONS:-2000}" \
  --agent.logger tensorboard \
  --log-root "../outputs/train"
