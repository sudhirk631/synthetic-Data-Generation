#!/usr/bin/env python3
"""
Helper to submit a compiled KFP pipeline YAML to a Kubeflow Pipelines instance.

Usage example:
  python gan_model/run_kfp.py --package gan_local_pipeline.yaml --image myuser/gan-train:latest --steps 200

If your KFP API is not at the default location, pass --host (e.g. http://127.0.0.1:8080).
"""
import argparse
from kfp import Client

def main():
    parser = argparse.ArgumentParser(description="Submit gan_local_pipeline.yaml to KFP")
    parser.add_argument("--package", required=True, help="Path to compiled pipeline YAML (e.g. gan_local_pipeline.yaml)")
    parser.add_argument("--image", required=True, help="Container image tag available to the cluster (e.g. myuser/gan-train:latest)")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--model-dir", default="./tmp/gan_model")
    parser.add_argument("--output-path", default="./final_generated.csv")
    parser.add_argument("--host", default=None, help="KFP API host (if needed), e.g. http://127.0.0.1:8080")
    parser.add_argument("--experiment", default="gan-local-demo")
    parser.add_argument("--run-name", default="gan-local-run")
    args = parser.parse_args()

    # Connect to KFP
    if args.host:
        client = Client(host=args.host)
    else:
        client = Client()  # assumes default environment / KFP client config

    # Pipeline parameters expected by gan_local_pipeline
    # Note: parameter names must match those declared in the pipeline function
    params = {
        "container_image": args.image,
        "training_steps": str(args.steps),
        "model_dir": args.model_dir,
        "output_path": args.output_path,
    }

    print(f"Submitting pipeline {args.package} to KFP (image={args.image})...")
    run = client.create_run_from_pipeline_package(
        pipeline_file=args.package,
        arguments=params,
        run_name=args.run_name,
        experiment_name=args.experiment,
    )

    # The returned object contains run metadata; print basic info
    try:
        run_id = run.run_id
    except AttributeError:
        # Older/newer kfp versions may return different shapes
        run_id = getattr(run, "id", None) or str(run)
    print("Pipeline submitted. Run id:", run_id)
    print("Open your KFP UI to monitor the run.")

if __name__ == "__main__":
    main()
