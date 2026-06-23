from __future__ import annotations

import argparse
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.limx_hud03.hud03_constants import HUD03_CONTROLLED_JOINTS, HUD03_PR_SPAWN_HEIGHT, HUD03_PR_XML


def joint_id(model: mujoco.MjModel, name: str) -> int:
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if index < 0:
    raise RuntimeError(f"Missing joint: {name}")
  return int(index)


def main() -> None:
  parser = argparse.ArgumentParser(description="Render a HU D03 NPZ reference motion.")
  parser.add_argument("--motion", type=Path, default=Path("motions/hud03_stand_dab_loop.npz"))
  parser.add_argument("--output", type=Path, default=Path("outputs/hud03_reference_motion.mp4"))
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--fps", type=int, default=30)
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(HUD03_PR_XML))
  data = mujoco.MjData(model)
  motion = np.load(args.motion, allow_pickle=True)
  joint_pos = motion["joint_pos"]
  source_fps = int(float(motion["fps"]))
  renderer = mujoco.Renderer(model, height=args.height, width=args.width)
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultCamera(camera)
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = (0.0, 0.0, 0.85)
  camera.distance = 2.8
  camera.azimuth = 90.0
  camera.elevation = -8.0

  frames = []
  stride = max(1, source_fps // args.fps)
  for index in range(joint_pos.shape[0]):
    qpos = model.qpos0.copy()
    qpos[:3] = (0.0, 0.0, HUD03_PR_SPAWN_HEIGHT)
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for joint_index, name in enumerate(HUD03_CONTROLLED_JOINTS):
      jid = joint_id(model, name)
      qpos[model.jnt_qposadr[jid]] = float(joint_pos[index, joint_index])
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    if index % stride == 0:
      renderer.update_scene(data, camera=camera)
      frames.append(renderer.render())
  args.output.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(args.output), frames, fps=args.fps)
  renderer.close()
  print(args.output)


if __name__ == "__main__":
  main()
