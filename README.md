# Synthetic-Data-Generation
The code sample in the repository is an implementation of synthetic data generation using GAN model. 
# Synthetic Data Generation System

A comprehensive system for generating synthetic data using Wasserstein GAN with Gradient Penalty (WGAN-GP) and an intelligent multi-agent system for natural language to SQL translation and data management.

---

## Table of Contents

1. [Overview](#overview)

2. [GAN Model Architecture (WGAN-GP)](#gan-model-architecture-wgan-gp)

3. [KFP Pipeline for Monthly Retraining](#kfp-pipeline-for-monthly-retraining)

4. [Multi-Agent System](#multi-agent-system)

5. [Project Structure](#project-structure)

6. [Setup & Installation](#setup--installation)

7. [Usage](#usage)

---

## Overview

Some of the common use cases of synthetic data generation are:
1. The data in Development or User Acceptance Test (UAT) are not sufficient enough in terms of variety, volume to test the product features.
2. The data from Production cannot be copied to lower environments due to compliance and data privacy issues.
3. It is a new product and data does not exist in production.
4. The data needs to be shared with universities or vendors for research for which production data cannot be shared.
5. In special cases generate data for Machine Leaning algorithms training, validation or testing.
   
Considering the above points, the initiatives was taken up to build an application to learn the patterns and be able to generate synthetic data. 
The capabilities which are coded are as:

- **Synthetic Data Generation**: Uses Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate high-quality synthetic datasets from real data.

- **Natural Language to SQL**: Converts user queries to BigQuery SQL using LLMs.

- **Intelligent Routing**: Routes requests to batch inference or direct database queries based on request size.

- **Monthly Retraining**: Automatically retrains the GAN model monthly using Google Cloud's Kubeflow Pipelines (KFP)
 
---

## GAN Model Architecture (WGAN-GP) 

### What is WGAN-GP?

**Wasserstein GAN with Gradient Penalty (WGAN-GP)** is an advanced generative model that improves upon standard GANs by:

- Using the Wasserstein distance as a loss metric (more stable training)
- Adding gradient penalty to enforce the Lipschitz constraint (prevents mode collapse)
- Generating more realistic and diverse synthetic data

### Model Component

Located in: [`gan_model/trainer/train1.py`](./gan_model/trainer/train1.py)

#### 1. **Generator**

```python

def create_generator_combined(num_cat, num_dt):

    # Input: Random noise (NOISE_DIM = 128)

    # Output: Synthetic data (categorical + datetime features)

    # Architecture:

    #   - Categorical branch: Dense(128) -> LeakyReLU -> BatchNorm -> Dense(num_cat, sigmoid)

    #   - Datetime branch: Dense(128) -> LeakyReLU -> BatchNorm -> Dense(num_dt, sigmoid)

    #   - Concatenate both branches for combined output

```

**Purpose**: Generates synthetic categorical and datetime features that closely mimic real data distribution.

 

#### 2. **Critic (Discriminator)**

```python

def create_critic_combined(num_cat, num_dt):

    # Input: Data (real or fake)

    # Output: Scalar score (not probability)

    # Architecture:

    #   - Dense(256) -> LeakyReLU -> Dropout(0.3)

    #   - Dense(128) -> LeakyReLU -> Dropout(0.3)

    #   - Dense(1) [score, not sigmoid]

```
**Purpose**: Distinguishes between real and synthetic data using Wasserstein distance.

#### 3. **Loss Functions**

- **Critic Loss** (with Gradient Penalty):

  ```

  L_critic = E[critic(fake)] - E[critic(real)] + λ * gradient_penalty

  ```

  where λ = 10 (penalty weight)

 
- **Generator Loss**:

  ```

  L_gen = -E[critic(fake)]

  ```

- **Gradient Penalty**:

  Enforces the Lipschitz constraint by penalizing gradients that deviate from norm 1.

### Data Transformers

#### 1. **CategoricalTransformer**

- Maps categorical values to continuous ranges based on frequency
- Uses fuzzy transformation with normal distribution sampling
- Reverses transformation to recover original categories

#### 2. **DatetimeTransformer**

- Converts datetime strings to integer timestamps (nanoseconds)
- Applies min-max scaling
- Handles null values with configurable fill strategies (mean, mode, or custom value)

#### 3. **NullTransformer (Optional)**

- Manages features with high null rates (>70%)
- Creates indicator columns for missing data
- Preserves null patterns in synthetic data

### Training Process

```python

def train_step(real_data, gen, crit, g_optimizer, d_optimizer):

    # 1. Critic updates (5x per generator update for stability)

    #    - Add noise to real data

    #    - Compute Wasserstein distance with gradient penalty

    #    - Update critic weights
   

    # 2. Generator update

    #    - Generate fake data from random noise

    #    - Clip values to [0, 1]

    #    - Update generator weights to fool critic


    # Training runs for N steps (default: 2000)

    # Batch size: 64

    # Optimizer: Adam (lr=0.0002, beta_1=0.5)

``` 

### Model Serving

- **Export Format**: TensorFlow SavedModel format

- **Storage**: Google Cloud Storage (GCS)

- **Location**: Specified via `--model-dir` parameter in training
---

## KFP Pipeline for Monthly Retraining 

Located in: [`gan_model/kfp_pipeline.py`](./gan_model/kfp_pipeline.py)

### Pipeline Overview

The pipeline automates monthly GAN retraining using Google Cloud's Kubeflow Pipelines (KFP). 

### Pipeline Stages
 
```

┌─────────────────────────────────────┐

│  Custom Training Job (GAN Trainer)  │  Runs train1.py in container

└──────────────┬──────────────────────┘

               │

               ▼

┌─────────────────────────────────────┐

│  Model Importer                     │  Imports trained model from GCS

└──────────────┬──────────────────────┘

               │

               ▼

┌─────────────────────────────────────┐

│  Model Upload to Vertex AI          │  Registers model for serving

└──────────────┬──────────────────────┘

               │

               ▼

        (Optional: Deploy to Endpoint)

```

### Configuration

**Project Settings**:

```python

PROJECT_ID = "project_id"

REGION = "region_id"

PIPELINE_ROOT = "gs://gan_test_1/pipeline_root"

```

**Default Parameters**:

- **Container**: `region_id-docker.pkg.dev/.../gan-training-container:latest`

- **Machine Type**: `n1-standard-4` (4 vCPUs, 15GB memory)

- **Training Steps**: 2000

- **Model Output**: `gs://gan_test_1/tmp/gan_model`

- **Synthetic Data Output**: `gs://gan_test_1/final_generated.csv`

### Scheduling

For testing purposes we have defined this as the cron job.

```

37 17 2 1 *  (minute hour day month day_of_week)

```

### Running the Pipeline Manually

```bash

python gan_model/kfp_pipeline.py 

# This:

# 1. Compiles pipeline to YAML (gan_packaged_pipeline.yaml)

# 2. Submits job to Vertex AI

# 3. Creates monthly schedule

```
 
### Pipeline Outputs

1. **Trained Model**: Saved to GCS at `{model_dir}`

    - Format: TensorFlow SavedModel

    - Contains generator and critic weights

2. **Synthetic Dataset**: CSV at `{output_path}`

    - Synthetic data from 1000 samples

    - Matches original schema
 
3. **Quality Report**: Logged to pipeline output

    - SDMetrics QualityReport
    - Column shape similarity scores

---

## Multi-Agent System

Located in: [`Synthetic_Data/`](./Synthetic_Data/)

### Agent Architecture
The system uses a two-tier agent architecture:
 
```

┌────────────────────────────────────────┐

│      Root Agent (db_ds_multiagent)     │

│  - Coordinates multiple sub-agents     │

│  - Routes requests intelligently       │

│  - Manages synthetic data workflows    │

└────┬─────────────────────────────────┬─┘

     │                                 │

     ▼                                 ▼

┌─────────────────────┐      ┌──────────────────┐

│ Database Agent      │      │ Batch Inference  │

│ (NL2SQL Translator) │      │ Agent            │

│                     │      │                  │

│ - Converts queries  │      │ - Generates      │

│ - Validates SQL     │      │   synthetic data │

│ - Executes queries  │      │ - Large batches  │

└─────────────────────┘      └──────────────────┘

```
 
### Entry Point Logic

Located in: [`Synthetic_Data/agent_setup.py`](./Synthetic_Data/agent_setup.py)

#### Root Agent Decision Flow 

```python

@dsl.component

def root_agent():

    # 1. Parse user request (question)

    # 2. Determine request size/type

    # 3. Route to appropriate sub-agent:


    if is_synthetic_data_request(question):

        if request_size < THRESHOLD:  # Direct DB query

            return call_db_agent(question)

        else:  # Large batch

            return batch_inferencing(question)

    else:  # Regular SQL analytics

        return call_db_agent(question)

``` 

#### Request Routing Logic

| Request Type | Request Size | Route | Tool |

|---|---|---|---|

| Synthetic Data | Small (< 10K rows) | Direct Database | `call_db_agent()` |

| Synthetic Data | Large (≥ 10K rows) | Batch Inference | `batch_inferencing()` |

| SQL Analytics | Any | Database | `call_db_agent()` |

| Data Management | Any | CSV Save | `save_to_csv()` |


### Sub-Agents 

#### 1. **Database Agent** (NL2SQL)

Located in: [`Synthetic_Data/agent_setup.py`](./Synthetic_Data/agent_setup.py#L23)

 
**Tools**:

- `initial_bq_nl2sql`: Converts natural language to SQL

- `run_bigquery_validation`: Validates and executes SQL

**Workflow**:

```

User Query → NL2SQL Translation → SQL Validation → BigQuery Execution → Results

```

**Example**:

```

User: "How many passengers booked flights in January?"

↓

Agent: "SELECT COUNT(*) FROM flights WHERE MONTH(booking_date) = 1"

↓

Tool: Validates SQL against BigQuery schema

↓

Tool: Executes query

↓

Returns: 42,537 passengers

```

#### 2. **Batch Inference Agent**

Located in: [`Synthetic_Data/tools.py`](./Synthetic_Data/tools.py) 

**Function**: `batch_inferencing(request: str, tool_context: ToolContext)`

**Workflow**:

```

Large Request → Vertex AI Batch Prediction → Process Results → Save CSV → Return Path

```

**Use Cases**:

- Generate > 10,000 synthetic records

- Bulk data masking operations

- Large-scale synthetic dataset creation

### System Components

#### LLM Module

File: [`Synthetic_Data/llm_utils.py`](./Synthetic_Data/llm_utils.py)

**GeminiModel Class**:

- Wrapper around Google Gemini LLM

- Retry logic with exponential backoff

- Parallel batch processing

- Response caching

#### SQL Translator

File: [`Synthetic_Data/sql_translator.py`](./Synthetic_Data/sql_translator.py)


**SqlTranslator Class**:

- Converts SQL dialects (SQLite → BigQuery)

- Auto-corrects invalid SQL

- Extracts table schema

- Uses SQLGlot for parsing 

#### Prompts

File: [`Synthetic_Data/prompts.py`](./Synthetic_Data/prompts.py)

**Includes**:

- `return_instructions_root()`: Root agent instructions

- `return_instructions_bigquery()`: Database agent instructions

- Divide-and-conquer prompts for complex queries

- Few-shot examples for SQL generation

#### BigQuery Utils

File: [`Synthetic_Data/bigquery_utils.py`](./Synthetic_Data/bigquery_utils.py) 

**Functions**:

- `get_bq_database_settings()`: Fetches schema and metadata

- `run_bigquery_validation()`: Validates and executes SQL

- Schema extraction with sample rows

#### Tools

File: [`Synthetic_Data/tools.py`](./Synthetic_Data/tools.py)

**Available Tools**:

1. `initial_bq_nl2sql`: NL to SQL translation

2. `save_to_csv`: Export results to GCS CSV

3. `batch_inferencing`: Large-scale batch processing

4. `call_db_agent`: Invoke database sub-agent

---

## Project Structure 

```

data-masking/

│

├── gan_model/

│   ├── trainer/

│   │   └── train1.py                 # GAN trainer with WGAN-GP

│   ├── kfp_pipeline.py               # Monthly retraining pipeline

│   └── Dockerfile                     # Container image for training

│

├── Synthetic_Data/

│   ├── __init__.py

│   ├── GAN_Combined.py               # Main orchestration (refactored)

│   ├── agent_setup.py                # Agent definitions

│   ├── llm_utils.py                  # LLM wrapper & utilities

│   ├── sql_translator.py             # SQL translation logic

│   ├── prompts.py                    # Prompt templates

│   ├── bigquery_utils.py             # BigQuery integration

│   ├── tools.py                      # Agent tools

│   ├── constants.py                  # Configuration constants

│   └── transformers.py               # Data transformers

│

├── pyproject.toml                    # Poetry dependency management

└── README.md                         # This file

```
---

## Setup & Installation

### Prerequisites 

- Python 3.9+

- Google Cloud Account with:

    - BigQuery enabled

    - Vertex AI enabled

    - Cloud Storage enabled

    - Service account with appropriate permissions

### Installation

1. **Clone the repository**:

   ```bash

   git clone https://github.com/sabre-internal/dap-cdpe.data-masking

   ```

2. **Set up Google Cloud authentication**:

   ```bash

   gcloud auth configure-docker us-docker.pkg.dev

   gcloud auth application-default login

   ```

3. **Configure environment variables**:

   ```bash

   export GOOGLE_CLOUD_PROJECT="sab-dev-dap-aimlpipeline-4474"

   export BIGQUERY_AGENT_MODEL="gemini-1.5-pro"

   export ROOT_AGENT_MODEL="gemini-1.5-pro"

   export GOOGLE_GENAI_USE_VERTEXAI=1

   ```

---

## Usage

### 1. Train GAN Model Manually (Has Default Params)

```bash

cd gan_model/trainer

python train1.py \

    --query "SELECT * FROM dataset.table" \

    --model-dir "gs://your-bucket/gan_model" \

    --output-path "gs://your-bucket/synthetic.csv" \

    --steps 2000

```

**Parameters**:

- `--query`: BigQuery SQL for training data

- `--model-dir`: GCS path to save trained model

- `--output-path`: GCS path for synthetic data CSV

- `--steps`: Number of training iterations


### 2. Deploy KFP Pipeline 

```bash

cd gan_model

python kfp_pipeline.py

```

This will:

- Compile the pipeline to YAML

- Submit it to Vertex AI Pipelines

- Create a monthly schedule (1st of each month, 5:37 PM UTC)

### 3. Run the ADK Agent

Run this command from the root of the repository.

```bash

adk web

```

This will:

- Trigger the ADK Agent which will be launched on your localhost for testing

- Deploy to Agent Engine/ Cloud Run for company-wide access

--- 

## Performance & Monitoring

### GAN Training Metrics

Monitor during training:

```

Step: 100, Critic Loss: -0.45, Generator Loss: -0.52, GP: 0.03

Step: 200, Critic Loss: -0.38, Generator Loss: -0.48, GP: 0.02

...

Step: 2000, Critic Loss: -0.25, Generator Loss: -0.30, GP: 0.01

``` 

**Key Indicators**:

- **Critic Loss**: Should stabilize around -0.2 to -0.3

- **Generator Loss**: Should stabilize around -0.2 to -0.3

- **Gradient Penalty**: Should remain close to 0


### Data Quality 

Quality reports generated using SDMetrics:

```

Column Shapes Score: 74.03%

Column Pairs Score: 68.51%

Overall Score: 71.27%

```

**Threshold**: Aim for ≥ 70% column shape similarity.

--- 

## Troubleshooting

### Authentication Issues

```bash

# Fix Docker authentication

gcloud auth configure-docker us-docker.pkg.dev

# Fix gcloud auth

gcloud auth application-default login

``` 

### BigQuery Connection Issues

- Verify service account has `roles/bigquery.user`

- Check dataset permissions

- Ensure queries use fully qualified table names

### Model Training Fails

- Check training data quality

- Verify null handling in data transformers

- Increase training steps if loss isn't converging

### Pipeline Scheduling

- Verify Vertex AI Pipelines API is enabled

- Check IAM permissions for service account

- Review cron expression: `37 17 2 1 *`

---
## References

- [WGAN-GP Paper](https://arxiv.org/abs/1704.00028)

- [Google Cloud Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)

- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)

- [SDMetrics Documentation](https://docs.sdmetrics.io/)
