# LIMX HU D03 mjlab Custom Mimic

Clean GitHub version of a LIMX HU D03 custom mimic project built on mjlab and the official HU D03 MuJoCo description.

This repository does **not** include copied sample mimic checkpoints, sample policy weights, training logs, or assignment-specific report outputs. The included workflow generates a custom forward-kinematics-consistent reference motion and trains a policy from scratch or from your own checkpoint.

## Demo videos

### Locomotion policy playback

<video src="assets/videos/locomotion-walk-0-50ms.mp4" controls muted loop width="720"></video>

[Open locomotion video](assets/videos/locomotion-walk-0-50ms.mp4)

### Custom mimic policy playback

<video src="assets/videos/mimic-policy-short-run.mp4" controls muted loop width="720"></video>

[Open mimic video](assets/videos/mimic-policy-short-run.mp4)

The mimic video is a short-run custom policy attempt trained on a generated HU D03 reference motion. It is included as a visual result, not as a copied pretrained checkpoint.

## What is included

- `mjlab/` – mjlab source with LIMX HU D03 velocity and tracking task integration.
- `humanoid-description-master/` – HU D03 MuJoCo model and meshes used by the project.
- `tools/generate_stand_dab_loop_motion.py` – generates the clean cyclic mimic motion.
- `tools/render_reference_motion.py` – renders the generated reference motion.
- `tools/train_mimic.sh` – trains the HU D03 mimic policy.
- `tools/render_policy.sh` – renders a trained policy checkpoint.
- `assets/videos/` – MP4 demo videos displayed in this README.

## What was intentionally removed

- Sample runtime repository copies.
- Pretrained sample checkpoints such as dance, jump, or locomotion weights.
- Assignment-specific logs, screenshots, PDFs, and phase notebooks.
- Large temporary training videos outside `assets/videos/`.
- Temporary Colab outputs and local training outputs.

## Motion sequence

The clean custom motion is:

```text
stand -> dab hold -> return stand -> repeat
```

The motion is generated procedurally from HU D03 joint poses and then converted into a tracking motion file using MuJoCo forward kinematics. The generated NPZ contains joint states and body states from the same HU D03 XML used for training.

## Setup

```bash
cd mjlab
uv sync
```

## Generate the reference motion

```bash
cd mjlab
uv run python ../tools/generate_stand_dab_loop_motion.py   --output ../motions/hud03_stand_dab_loop.npz   --mapping ../outputs/hud03_stand_dab_loop_mapping.csv   --gate ../outputs/hud03_motion_gate.json
```

The motion gate must return `PASS` before training.

## Render the reference motion

```bash
cd mjlab
uv run python ../tools/render_reference_motion.py   --motion ../motions/hud03_stand_dab_loop.npz   --output ../outputs/hud03_reference_motion.mp4   --width 640   --height 480
```

## Train the custom mimic policy

```bash
HUD03_NUM_ENVS=256 HUD03_MAX_ITERATIONS=2000 tools/train_mimic.sh
```

For longer training, increase `HUD03_MAX_ITERATIONS`. The original short-run demo used 2000 iterations for a quick custom policy, but a higher quality policy should use longer training.

## Render a trained policy

```bash
tools/render_policy.sh checkpoints/model_1999.pt
```

The script expects your own checkpoint path. No pretrained sample checkpoint is included.

## Important environment flags

- `HUD03_EASY_MIMIC=1` enables the stable short-run mimic configuration.
- `HUD03_DAB_STRICT=1` increases upper-body tracking rewards for the dab motion.

These flags replace the older assignment-specific naming and are safe for a clean public repository.

## Notes

This project is a custom HU D03 mimic-training workflow. It uses a generated stand-dab-return loop motion and does not copy sample mimic policy weights from external repositories.
