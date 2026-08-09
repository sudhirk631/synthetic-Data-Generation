from kfp import dsl
from kfp.compiler import Compiler

import os
import subprocess
import argparse
from pathlib import Path

# This file provides a Kubeflow Pipeline that is runnable on a local KFP installation
# (e.g. Kubeflow Pipelines on Minikube / Kind) and also a simple "run locally" mode
# that executes the training step directly on your laptop without any cloud dependencies.
# It removes Google Cloud Pipeline Components and Vertex AI usage.

# Defaults suitable for local demo
DEFAULT_CONTAINER_IMAGE = "python:3.9-slim"
DEFAULT_MODEL_DIR = "./tmp/gan_model"
DEFAULT_OUTPUT_PATH = "./final_generated.csv"

@dsl.pipeline(
    name="gan-local-training-pipeline",
    description="A simplified KFP pipeline for a local/demo environment (no Vertex AI).",
    pipeline_root="./pipeline_root",
)
def gan_local_pipeline(
        display_name: str = "gan-training-local",
        container_image: str = DEFAULT_CONTAINER_IMAGE,
        model_dir: str = DEFAULT_MODEL_DIR,
        output_path: str = DEFAULT_OUTPUT_PATH,
        training_steps: int = 2000
):
    """
    Pipeline that runs the training container as a ContainerOp. This assumes you have
    a runnable training entrypoint inside the container (for example `python -m gan_model.train`).

    To run this pipeline locally you need a Kubeflow Pipelines installation (e.g. using
    Minikube or Kind) and a container registry accessible from that cluster (or use a
    local image builder in the cluster).
    """

    train = dsl.ContainerOp(
        name="train-gan",
        image=container_image,
        command=["python", "-m", "gan_model.train"],
        arguments=[
            "--steps", str(training_steps),
            "--model-dir", model_dir,
            "--output-path", output_path,
        ],
    )

    # If you want a second step to consume the result you can add another ContainerOp
    # that depends on `train` and reads `output_path`.


def run_training_locally(training_steps: int, model_dir: str, output_path: str):
    """
    Simple local runner for demo purposes. This runs the training entrypoint on the
    local machine (no KFP). It tries the following in order:
      1. `python -m gan_model.train` (preferred)
      2. `gan_model/train.py` script if present

    Adjust this function if your training entrypoint differs (for example if you want
    to use Docker locally).
    """
    model_dir_path = Path(model_dir)
    model_dir_path.parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    module_cmd = ["python", "-m", "gan_model.train",
                  "--steps", str(training_steps),
                  "--model-dir", model_dir,
                  "--output-path", output_path]
    try:
        print("Running local training via module:", " ".join(module_cmd))
        subprocess.check_call(module_cmd)
        return
    except subprocess.CalledProcessError:
        print("Module run failed, trying script fallback...")

    script = Path("gan_model/train.py")
    if script.exists():
        script_cmd = ["python", str(script),
                      "--steps", str(training_steps),
                      "--model-dir", model_dir,
                      "--output-path", output_path]
        print("Running local training via script:", " ".join(script_cmd))
        subprocess.check_call(script_cmd)
        return

    raise RuntimeError(
        "Could not find a training entrypoint. Create `gan_model/train.py` or make the package runnable as `python -m gan_model.train`, or update run_training_locally() to call your desired command.`"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run the training step locally (no KFP).")
    parser.add_argument("--compile", action="store_true", help="Compile the KFP pipeline to a YAML file (gan_local_pipeline.yaml).")
    parser.add_argument("--image", type=str, default=DEFAULT_CONTAINER_IMAGE, help="Container image to use for the ContainerOp when compiling/running on KFP.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    if args.local:
        print("Running training locally (no KFP).")
        run_training_locally(training_steps=args.steps, model_dir=args.model_dir, output_path=args.output_path)
        print("Local training finished. Model dir:", args.model_dir, "Output:", args.output_path)
        raise SystemExit(0)

    if args.compile:
        print("Compiling pipeline to gan_local_pipeline.yaml")
        Compiler().compile(
            pipeline_func=gan_local_pipeline,
            package_path="gan_local_pipeline.yaml"
        )
        print("Compiled. You can upload gan_local_pipeline.yaml to your Kubeflow Pipelines UI or use the KFP SDK to submit it.")
        raise SystemExit(0)

    # Default behavior when neither --local nor --compile is passed: compile the pipeline
    # so users get a YAML by running the script with no args (keeps behaviour similar to the original file).
    Compiler().compile(
        pipeline_func=gan_local_pipeline,
        package_path="gan_local_pipeline.yaml"
    )
    print("Generated gan_local_pipeline.yaml. To run locally (no KFP), use --local. To compile only use --compile.")
