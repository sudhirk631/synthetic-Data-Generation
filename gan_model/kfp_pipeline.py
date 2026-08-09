from kfp import dsl
from kfp.compiler import Compiler
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google.cloud import aiplatform
from google_cloud_pipeline_components.types import artifact_types

import os
import subprocess
import argparse
from pathlib import Path

PROJECT_ID = "project_id"
REGION = "region_id"
PIPELINE_ROOT = "gs://gan_test_1/pipeline_root"

@dsl.pipeline(
    name="gan-packaged-training-pipeline",
    description="A pipeline that runs a packaged GAN trainer as a custom job.",
    pipeline_root=PIPELINE_ROOT,
)
def gan_packaged_pipeline(
        display_name: str = "gan-training-from-package",
        container_uri: str = "region_id-docker.pkg.dev/project_id/synthetic-data-generator/gan-training-container:latest",
        machine_type: str = "n1-standard-4",
        model_dir: str = "gs://gan_test_1/tmp/gan_model",
        output_path: str = "gs://gan_test_1/final_generated.csv",
        training_steps: int = 2000
):
    # NOTE: This function is unchanged for the GCP/KFP path. For local runs we bypass pipeline creation
    custom_job_task = CustomTrainingJobOp(
        project=PROJECT_ID,
        location=REGION,
        display_name=display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_uri,
                    "args": [
                        f"--steps={training_steps}",
                        f"--model-dir={model_dir}",
                        f"--output-path={output_path}",
                    ],
                },
            }
        ],
    )

    model_importer = dsl.importer(
        artifact_uri=model_dir,
        artifact_class=artifact_types.UnmanagedContainerModel,
        metadata={
            "containerSpec": {
                "imageUri": container_uri
            }
        }
    )
    model_importer.after(custom_job_task)

    model_upload_task = ModelUploadOp(
        project=PROJECT_ID,
        location=REGION,
        display_name="gan-model",
        description="gan-model",
        unmanaged_container_model=model_importer.output,
        labels={},
    )
    model_upload_task.after(model_importer)

    # Endpoint creation / deploy commented out in original; left out for brevity.

def run_training_locally(training_steps: int, model_dir: str, output_path: str, container_uri: str = None):
    """
    Run training locally for demo:
    - If you have a local Python training script, call it (preferred).
    - Or, if you use a module entrypoint, call `python -m gan_model.train`.
    - If you built a local Docker image and want to run it, replace the command below.
    """
    # Ensure local paths exist
    model_dir_path = Path(model_dir)
    model_dir_path.parent.mkdir(parents=True, exist_ok=True)
    output_path_parent = Path(output_path).parent
    output_path_parent.mkdir(parents=True, exist_ok=True)

    # Try to run a Python module entrypoint; edit to match your actual training script if needed.
    # Preferred: have a module gan_model.train with a main() or runnable as `python -m gan_model.train`
    python_cmd = ["python", "-m", "gan_model.train",
                  f"--steps={training_steps}",
                  f"--model-dir={model_dir}",
                  f"--output-path={output_path}"]
    try:
        print("Running local training with:", " ".join(python_cmd))
        subprocess.check_call(python_cmd)
    except subprocess.CalledProcessError:
        # Fallback: try to run a script file if available
        script = Path("gan_model/train.py")
        if script.exists():
            cmd = ["python", str(script),
                   f"--steps={training_steps}",
                   f"--model-dir={model_dir}",
                   f"--output-path={output_path}"]
            print("Falling back to script:", " ".join(cmd))
            subprocess.check_call(cmd)
        else:
            # If neither module nor script exists, tell the user what to change
            raise RuntimeError(
                "Could not run training locally — no module 'gan_model.train' runnable and no gan_model/train.py found. "
                "Either create a train.py or replace run_training_locally() with the appropriate command (e.g. docker run ...)."
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run training locally for demo (no KFP/GCP).")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--model-dir", type=str, default="./tmp/gan_model")
    parser.add_argument("--output-path", type=str, default="./final_generated.csv")
    parser.add_argument("--container-uri", type=str, default="region_id-docker.pkg.dev/project_id/synthetic-data-generator/gan-training-container:latest")
    args = parser.parse_args()

    # Also respect environment variable fallback
    local_mode = args.local or os.environ.get("LOCAL_DEMO") == "1"

    if local_mode:
        # Local demo: run training directly on the laptop and exit
        print("Running in LOCAL demo mode. Training will run locally (no KFP/GCP).")
        run_training_locally(training_steps=args.steps, model_dir=args.model_dir, output_path=args.output_path, container_uri=args.container_uri)
        print("Local training finished. Model dir:", args.model_dir, "Output file:", args.output_path)
    else:
        # Original KFP/GCP flow: compile template and create a scheduled PipelineJob on Vertex AI
        Compiler().compile(
            pipeline_func=gan_packaged_pipeline,
            package_path="gan_packaged_pipeline.yaml"
        )

        aiplatform.init(project=PROJECT_ID, location=REGION)
        pipeline_job = aiplatform.PipelineJob(
            display_name="gan-packaged-training-pipeline",
            template_path="gan_packaged_pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            parameter_values={
                "training_steps": args.steps,
                "container_uri": args.container_uri,
                "machine_type": "n1-standard-4",
                "model_dir": "gs://gan_test_1/tmp/gan_model",
                "output_path": "gs://gan_test_1/final_generated_test.csv"
            }
        )

        job_schedule = aiplatform.PipelineJobSchedule(
            pipeline_job=pipeline_job,
            display_name="monthly-gan-training-schedule"
        )

        # NOTE: keep scheduling only when actually on GCP; this will error locally.
        job_schedule.create("37 17 2 1 *")
        print("Monthly pipeline schedule created successfully.")
