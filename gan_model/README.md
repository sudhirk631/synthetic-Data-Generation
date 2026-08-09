# GAN model — local KFP demo

This folder contains a local-friendly Kubeflow Pipelines (KFP) pipeline and a small helper script to submit the compiled pipeline to a KFP instance.

## Files

- `pyproject.toml` — Project configuration and dependencies (modern Python packaging format)
- `kfp_pipeline.py` — pipeline that can be compiled to `gan_local_pipeline.yaml` and also supports `--local` to run training without KFP.
- `run_kfp.py` — helper to submit the compiled YAML to a KFP instance.
- `train.py` — training entrypoint (already present in the repo). The pipeline and local-run expect the command `python -m gan_model.train` or `gan_model/train.py`.

## Installation

Install the package with all dependencies:

```bash
pip install -e .
```

This reads dependencies from `pyproject.toml` and installs the package in editable mode.

## Quick local demo (no KFP)

1. Run:
   ```
   python gan_model/kfp_pipeline.py --local --steps 100
   ```
   This invokes your local training entrypoint and writes `./final_generated.csv` by default.

## Compile pipeline YAML

1. Compile the pipeline:
   ```
   python gan_model/kfp_pipeline.py --compile
   ```
   This produces `gan_local_pipeline.yaml`.

## Run on local Kubeflow Pipelines (Minikube / Kind)

1. Build an image with your code:
   ```
   docker build -t myuser/gan-train:latest .
   ```
2. Make the image available to the cluster:
   - For Minikube:
     ```
     minikube image load myuser/gan-train:latest
     ```
   - Or push to a registry the cluster can access:
     ```
     docker push myuser/gan-train:latest
     ```
3. Compile the YAML if not already:
   ```
   python gan_model/kfp_pipeline.py --compile
   ```
4. Submit via helper script:
   ```
   python gan_model/run_kfp.py --package gan_local_pipeline.yaml --image myuser/gan-train:latest --steps 200 --host http://127.0.0.1:8080
   ```
   (Omit --host if Client() connects to your KFP automatically.)

## Notes

- The pipeline's ContainerOp runs `python -m gan_model.train` inside the image — ensure the image contains the package and entrypoint.
- For quick demos, prefer `--local` which requires no Docker/minikube setup.
- Dependencies are managed in `pyproject.toml`. The legacy `setup.py` can be deleted.
