from __future__ import annotations

from typing import Optional, Sequence

from benchmark.surgwmbench import BaselineSpec, run_cli


SPEC = BaselineSpec(
    baseline="surgsora",
    model="SurgSora",
    native_entrypoint="gradio_demo_run.py",
    native_train_entrypoint="Training/train_stage2.py",
    native_frame_predictor=(
        "upstream release exposes a Gradio demo and two-stage training scripts; a "
        "non-interactive future-prediction command still needs model-specific wiring."
    ),
    notes=(
        "SurgSora is object/trajectory-control oriented. The SurgWMBench adapter "
        "validates official manifests and trajectory targets, while native execution "
        "requires DAV2/CMP/SAM/controlnet assets and a non-interactive wrapper."
    ),
)


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_cli(SPEC, argv)


if __name__ == "__main__":
    main()
