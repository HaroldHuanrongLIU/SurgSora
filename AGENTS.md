# Repository Guidelines

## Project Structure & Module Organization

This repository contains SurgSora, an object-aware diffusion system for controllable surgical video generation. The root inference path uses `gradio_demo_run.py`, `pipeline/`, `models/`, and helpers in `utils/`. Training code lives under `Training/`, with model definitions in `Training/models/`, data utilities in `Training/train_utils/`, and entrypoints `Training/train_stage1.py` and `Training/train_stage2.py`. Demo assets are in `assets/` and `demo/`. Large checkpoints belong under `models/` or `Training/ckpts/` and should not be committed.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install project dependencies.
- `git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .`: install SAM2 outside this repo.
- `python gradio_demo_run.py`: launch the local Gradio demo from the repository root after checkpoints are available.
- `cd Training && bash train_stage1.sh`: run stage 1 training; confirm checkpoint paths match your layout.
- `cd Training && bash train_stage2.sh`: run stage 2 training after the stage 1 ControlNet checkpoint exists.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and group imports as standard library, third-party, then local modules. Match existing naming: lowercase files for pipelines and utilities (`pipeline.py`, `flow_viz.py`) and CamelCase class names for model components (`DualFlowControlNet`, `CMP_demo`). Keep explicit tensor shape variables such as `H`, `W`, and `num_frames` where surrounding code uses them. No formatter or linter config is checked in; keep changes locally formatted and avoid broad style-only rewrites.

## Testing Guidelines

There is no formal test suite in this snapshot. For inference or training changes, run the narrowest executable smoke check: import the touched module, launch the relevant script with minimal settings when practical, or run the Gradio app far enough to verify model loading. Add future tests under top-level `tests/` using `test_*.py` names when introducing reusable data or utility logic.

## Commit & Pull Request Guidelines

Recent history uses short, plain subjects such as `Update README.md` and `Add files via upload`. For new work, use concise imperative subjects that name the change, for example `Add SurgWMBench data loader` or `Fix Gradio checkpoint paths`. Pull requests should include a summary, commands or smoke checks run, required checkpoint or dataset paths, and screenshots or sample links for UI or video-output changes.

## Security & Configuration Tips

Do not commit downloaded weights, generated videos, local datasets, credentials, or machine-specific absolute paths. Document required checkpoint locations in README updates or script comments, and keep defaults relative to the repository root whenever possible.
