from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.limx_hud03.hud03_constants import (
  HUD03_CONTROLLED_JOINTS,
  HUD03_PR_HOME_JOINT_POS,
  HUD03_PR_SPAWN_HEIGHT,
  HUD03_PR_XML,
)


def joint_id(model: mujoco.MjModel, name: str) -> int:
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if index < 0:
    raise RuntimeError(f"Missing joint: {name}")
  return int(index)


def body_names(model: mujoco.MjModel) -> list[str]:
  names = []
  for index in range(1, model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
    names.append(name or f"body_{index}")
  return names


def smoothstep(x: float) -> float:
  x = float(np.clip(x, 0.0, 1.0))
  return x * x * (3.0 - 2.0 * x)


def blend_pose(home: dict[str, float], target: dict[str, float], alpha: float) -> dict[str, float]:
  pose = dict(home)
  for name, value in target.items():
    pose[name] = (1.0 - alpha) * pose[name] + alpha * value
  return pose


def dab_alpha(t: float, cycle_seconds: float) -> float:
  local = float(t % cycle_seconds)
  if local < 0.45:
    return 0.0
  if local < 1.35:
    return smoothstep((local - 0.45) / 0.90)
  if local < 2.55:
    return 1.0
  if local < 3.55:
    return 1.0 - smoothstep((local - 2.55) / 1.00)
  return 0.0


def build_joint_pose(t: float, cycle_seconds: float) -> dict[str, float]:
  alpha = dab_alpha(t, cycle_seconds)
  local = float(t % cycle_seconds)
  hold_sway = 0.025 * np.sin(2.0 * np.pi * local / cycle_seconds) * alpha
  target = dict(HUD03_PR_HOME_JOINT_POS)
  target.update({
    "waist_yaw_joint": -0.16,
    "waist_roll_joint": 0.06,
    "waist_pitch_joint": 0.03,
    "head_yaw_joint": -0.30,
    "head_pitch_joint": -0.16,
    "left_shoulder_pitch_joint": -0.62,
    "left_shoulder_roll_joint": 0.92,
    "left_shoulder_yaw_joint": -0.36,
    "left_elbow_joint": -0.08,
    "left_wrist_yaw_joint": 0.20,
    "left_wrist_pitch_joint": -0.08,
    "left_hand_yaw_joint": 0.08,
    "right_shoulder_pitch_joint": 0.62,
    "right_shoulder_roll_joint": -0.78,
    "right_shoulder_yaw_joint": 0.34,
    "right_elbow_joint": -0.94,
    "right_wrist_yaw_joint": -0.20,
    "right_wrist_pitch_joint": 0.08,
    "right_hand_yaw_joint": -0.08,
  })
  pose = blend_pose(HUD03_PR_HOME_JOINT_POS, target, alpha)
  pose["waist_yaw_joint"] += hold_sway
  pose["head_yaw_joint"] += 0.5 * hold_sway
  return pose


def clip_pose(model: mujoco.MjModel, pose: dict[str, float]) -> tuple[dict[str, float], list[dict]]:
  clipped = dict(pose)
  violations = []
  for name in HUD03_CONTROLLED_JOINTS:
    jid = joint_id(model, name)
    if model.jnt_limited[jid]:
      low, high = model.jnt_range[jid]
      raw = float(clipped[name])
      clipped[name] = float(np.clip(raw, low + 1.0e-5, high - 1.0e-5))
      if abs(raw - clipped[name]) > 1.0e-8:
        violations.append({"joint": name, "value": raw, "clipped": clipped[name], "low": float(low), "high": float(high)})
  return clipped, violations


def write_joint_qpos(model: mujoco.MjModel, qpos: np.ndarray, pose: dict[str, float]) -> None:
  for name in HUD03_CONTROLLED_JOINTS:
    jid = joint_id(model, name)
    qpos[model.jnt_qposadr[jid]] = pose[name]


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
  w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
  return np.stack([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], axis=-1)


def quat_inv(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[..., 1:] *= -1.0
  return out


def angular_velocity_from_quat(quat: np.ndarray, dt: float) -> np.ndarray:
  q = quat.astype(np.float64).copy()
  for i in range(1, q.shape[0]):
    flip = np.sum(q[i - 1] * q[i], axis=-1) < 0.0
    q[i, flip] *= -1.0
  dq = quat_mul(q[1:], quat_inv(q[:-1]))
  vec = dq[..., 1:]
  norm = np.linalg.norm(vec, axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(norm, np.clip(dq[..., :1], -1.0, 1.0))
  axis = np.divide(vec, np.maximum(norm, 1.0e-8))
  omega_pair = axis * angle / dt
  omega = np.zeros((q.shape[0], q.shape[1], 3), dtype=np.float32)
  omega[1:] = omega_pair.astype(np.float32)
  omega[0] = omega[1]
  return omega


def validate_motion(model: mujoco.MjModel, body_name_list: list[str], joint_pos: np.ndarray, body_pos: np.ndarray, body_quat: np.ndarray, body_lin_vel: np.ndarray, body_ang_vel: np.ndarray, violations: list[dict], fps: int, duration: float, cycle_seconds: float, cycle_count: int, output_path: Path, mapping_path: Path) -> dict:
  nonroot = slice(1, None)
  first_last_joint_error = float(np.max(np.abs(joint_pos[0] - joint_pos[-1])))
  metrics = {
    "joint_change": float(np.max(np.abs(joint_pos - joint_pos[0:1]))),
    "nonroot_body_pos_change": float(np.max(np.abs(body_pos[:, nonroot] - body_pos[0:1, nonroot]))),
    "nonroot_body_quat_change": float(np.max(np.abs(body_quat[:, nonroot] - body_quat[0:1, nonroot]))),
    "body_lin_vel_max": float(np.max(np.abs(body_lin_vel))),
    "body_ang_vel_max": float(np.max(np.abs(body_ang_vel))),
    "first_last_joint_error": first_last_joint_error,
    "finite": bool(np.isfinite(joint_pos).all() and np.isfinite(body_pos).all() and np.isfinite(body_quat).all() and np.isfinite(body_lin_vel).all() and np.isfinite(body_ang_vel).all()),
  }
  checks = {
    "joint_dim_31": joint_pos.shape[1] == 31,
    "body_dim_matches_xml": body_pos.shape[1] == model.nbody - 1,
    "joint_changes": metrics["joint_change"] > 0.10,
    "body_pos_changes": metrics["nonroot_body_pos_change"] > 0.04,
    "body_quat_changes": metrics["nonroot_body_quat_change"] > 0.04,
    "lin_vel_changes": metrics["body_lin_vel_max"] > 0.01,
    "ang_vel_changes": metrics["body_ang_vel_max"] > 0.01,
    "loop_returns_to_stand": first_last_joint_error < 1.0e-4,
    "finite": metrics["finite"],
  }
  return {
    "status": "PASS" if all(checks.values()) else "BLOCKED",
    "motion_path": str(output_path),
    "source_xml": str(HUD03_PR_XML),
    "sequence": "stand -> dab hold -> return stand -> repeat",
    "frames": int(joint_pos.shape[0]),
    "fps": int(fps),
    "cycle_seconds": float(cycle_seconds),
    "cycle_count": int(cycle_count),
    "duration_s": float(duration),
    "joint_dim": int(joint_pos.shape[1]),
    "body_dim": int(body_pos.shape[1]),
    "body_names_head": body_name_list[:10],
    "mapping_path": str(mapping_path),
    "metrics": metrics,
    "checks": checks,
    "limit_clip_count": len(violations),
    "limit_clip_examples": violations[:10],
    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
  }


def generate_motion(output_path: Path, mapping_path: Path, gate_path: Path, fps: int, cycle_seconds: float, cycle_count: int) -> dict:
  duration = cycle_seconds * cycle_count
  dt = 1.0 / fps
  model = mujoco.MjModel.from_xml_path(str(HUD03_PR_XML))
  data = mujoco.MjData(model)
  names = body_names(model)
  steps = int(duration * fps) + 1
  times = np.arange(steps, dtype=np.float32) * dt
  joint_pos = np.zeros((steps, len(HUD03_CONTROLLED_JOINTS)), dtype=np.float32)
  body_pos = np.zeros((steps, len(names), 3), dtype=np.float32)
  body_quat = np.zeros((steps, len(names), 4), dtype=np.float32)
  violations = []

  for index, t in enumerate(times):
    pose, clipped = clip_pose(model, build_joint_pose(float(t), cycle_seconds))
    violations.extend(clipped)
    qpos = model.qpos0.copy()
    qpos[:3] = (0.0, 0.0, HUD03_PR_SPAWN_HEIGHT)
    qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    write_joint_qpos(model, qpos, pose)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    joint_pos[index] = [pose[name] for name in HUD03_CONTROLLED_JOINTS]
    for body_index in range(1, model.nbody):
      body_pos[index, body_index - 1] = data.xpos[body_index]
      body_quat[index, body_index - 1] = data.xquat[body_index]

  joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
  body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
  body_ang_vel = angular_velocity_from_quat(body_quat, dt)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  mapping_path.parent.mkdir(parents=True, exist_ok=True)
  gate_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.array(fps, dtype=np.float32),
    duration=np.array(duration, dtype=np.float32),
    cycle_seconds=np.array(cycle_seconds, dtype=np.float32),
    cycle_count=np.array(cycle_count, dtype=np.int32),
    sequence=np.array("stand -> dab hold -> return stand -> repeat"),
    joint_names=np.array(HUD03_CONTROLLED_JOINTS),
    body_names=np.array(names),
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos,
    body_quat_w=body_quat,
    body_lin_vel_w=body_lin_vel,
    body_ang_vel_w=body_ang_vel,
    source_xml=np.array(str(HUD03_PR_XML)),
  )
  with mapping_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["index", "joint_name", "home_pose"])
    for index, name in enumerate(HUD03_CONTROLLED_JOINTS):
      writer.writerow([index, name, HUD03_PR_HOME_JOINT_POS[name]])
  payload = validate_motion(model, names, joint_pos, body_pos, body_quat, body_lin_vel, body_ang_vel, violations, fps, duration, cycle_seconds, cycle_count, output_path, mapping_path)
  gate_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
  return payload


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate a FK-consistent HU D03 stand-dab-return loop motion.")
  parser.add_argument("--output", type=Path, default=Path("motions/hud03_stand_dab_loop.npz"))
  parser.add_argument("--mapping", type=Path, default=Path("outputs/hud03_stand_dab_loop_mapping.csv"))
  parser.add_argument("--gate", type=Path, default=Path("outputs/hud03_motion_gate.json"))
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--cycle-seconds", type=float, default=4.0)
  parser.add_argument("--cycle-count", type=int, default=3)
  args = parser.parse_args()
  payload = generate_motion(args.output, args.mapping, args.gate, args.fps, args.cycle_seconds, args.cycle_count)
  print(json.dumps(payload, indent=2, ensure_ascii=False))
  if payload["status"] != "PASS":
    raise SystemExit("Motion validation blocked. Do not train this motion.")


if __name__ == "__main__":
  main()
