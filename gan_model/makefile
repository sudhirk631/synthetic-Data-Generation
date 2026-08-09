# Define variables

PROJECT_ID := sab-dev-dap-aimlpipeline-4474

REGION := us-central1

IMAGE_NAME := gan-training-container

IMAGE_URI := us-central1-docker.pkg.dev/$(PROJECT_ID)/synthetic-data-generator/$(IMAGE_NAME):latest

 

# New variables based on the example

ENGINE := docker

DOCKERFILE := Dockerfile

DOCKER_BUILD_ARGS := --build-arg IMAGE_REPO=repository.sabre-gcp.com:9083

DOCKER_IMAGE_NAME := $(IMAGE_NAME):latest

GAR_DOCKER_IMAGE_NAME := $(IMAGE_URI)

 

# Placeholder for prepare-build-outputs (add your pre-build logic here if needed)

prepare-build-outputs:

               @echo "Preparing build outputs..."

 

# Build the Docker image (updated to match example)

build-container: prepare-build-outputs

               $(ENGINE) build -f "$(DOCKERFILE)" $(DOCKER_BUILD_ARGS) -t "$(DOCKER_IMAGE_NAME)" .

 

# Push the image to Artifact Registry (updated to match example)

push-container-to-gar: build-container

               $(ENGINE) tag "$(DOCKER_IMAGE_NAME)" "$(GAR_DOCKER_IMAGE_NAME)"

               $(ENGINE) push "$(GAR_DOCKER_IMAGE_NAME)"

 

# Legacy build target (kept for compatibility)

build:

               docker build -t $(IMAGE_URI) .

 

# Legacy push target (kept for compatibility)

push: build

               docker push $(IMAGE_URI)

 

# Clean up local images

clean:

               docker rmi $(IMAGE_URI)

               docker rmi $(DOCKER_IMAGE_NAME)
