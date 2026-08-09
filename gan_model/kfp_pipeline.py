from kfp import dsl

from kfp.compiler import Compiler

from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

from google_cloud_pipeline_components.v1.model import ModelUploadOp

from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

from google.cloud import aiplatform

from google_cloud_pipeline_components.types import artifact_types

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


    # endpoint_task = EndpointCreateOp(

    #     project=PROJECT_ID,

    #     location=REGION,

    #     display_name="gan-endpoint"

    # )

    # endpoint_task.after(model_upload_task)

    #

    # deploy_task = ModelDeployOp(

    #     model=model_upload_task.outputs["model"],

    #     endpoint=endpoint_task.outputs["endpoint"],

    #     deployed_model_display_name="gan-deployed-model",

    #     machine_type=machine_type,

    #     traffic_split={"0": 100},

    #     project=PROJECT_ID,

    #     location=REGION,

    # )

    # deploy_task.after(endpoint_task)
 

if __name__ == "__main__":

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

            "training_steps": 2000,

            "container_uri": "region_id-docker.pkg.dev/project_id/synthetic-data-generator/gan-training-container:latest",

            "machine_type": "n1-standard-4",

            "model_dir": "gs://gan_test_1/tmp/gan_model",

            "output_path": "gs://gan_test_1/final_generated_test.csv"

        }

    )

    job_schedule = aiplatform.PipelineJobSchedule(

        pipeline_job=pipeline_job,

        display_name="monthly-gan-training-schedule"

    )

    job_schedule.create("37 17 2 1 *")


    print("Monthly pipeline schedule created successfully.")
