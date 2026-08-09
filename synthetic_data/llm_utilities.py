"""LLM utilities for the CHASE-SQL Agent."""

import functools

import os

import random

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Callable, List, Optional

 

import dotenv

import vertexai

from google.cloud import aiplatform

from vertexai.generative_models import (GenerationConfig, HarmBlockThreshold, HarmCategory)

from vertexai.preview import caching

from vertexai.preview.generative_models import GenerativeModel

 

dotenv.load_dotenv(override=True)

 

SAFETY_FILTER_CONFIG = {

    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,

    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,

    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,

    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,

}

 

GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

 

GEMINI_AVAILABLE_REGIONS = [

    "europe-west3", "australia-southeast1", "us-east4", "northamerica-northeast1",

    "europe-central2", "us-central1", "europe-north1", "europe-west8", "us-south1",

    "us-east1", "asia-east2", "us-west1", "europe-west9", "europe-west2",

    "europe-west6", "europe-southwest1", "us-west4", "asia-northeast1", "asia-east1",

    "europe-west1", "europe-west4", "asia-northeast3", "asia-south1",

    "asia-southeast1", "southamerica-east1",

]

 

GEMINI_URL = "projects/{GCP_PROJECT}/locations/{region}/publishers/google/models/{model_name}"

 

aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

 

 

def retry(max_attempts=8, base_delay=1, backoff_factor=2):

    """Decorator to add retry logic to a function."""

    def decorator(func):

        @functools.wraps(func)

        def wrapper(*args, **kwargs):

            attempts = 0

            while attempts < max_attempts:

                try:

                    return func(*args, **kwargs)

                except Exception as e:

                    print(f"Attempt {attempts + 1} failed with error: {e}")

                    attempts += 1

                    if attempts >= max_attempts:

                        raise e

                    delay = base_delay * (backoff_factor ** attempts)

                    delay = delay + random.uniform(0, 0.1 * delay)

                    time.sleep(delay)

        return wrapper

    return decorator

 

 

class GeminiModel:

    """Class for the Gemini model."""

 

    def __init__(

            self,

            model_name: str = "gemini-2.0-flash-001",

            finetuned_model: bool = False,

            distribute_requests: bool = False,

            cache_name: str | None = None,

            temperature: float = 0.01,

            **kwargs,

    ):

        self.model_name = model_name

        self.finetuned_model = finetuned_model

        self.arguments = kwargs

        self.distribute_requests = distribute_requests

        self.temperature = temperature

        model_name = self.model_name

        if not self.finetuned_model and self.distribute_requests:

            random_region = random.choice(GEMINI_AVAILABLE_REGIONS)

            model_name = GEMINI_URL.format(

                GCP_PROJECT=GCP_PROJECT,

                region=random_region,

                model_name=self.model_name,

            )

        if cache_name is not None:

            cached_content = caching.CachedContent(cached_content_name=cache_name)

            self.model = GenerativeModel.from_cached_content(cached_content=cached_content)

        else:

            self.model = GenerativeModel(model_name=model_name)

 

    @retry(max_attempts=12, base_delay=2, backoff_factor=2)

    def call(self, prompt: str, parser_func=None) -> str:

        """Calls the Gemini model with the given prompt."""

        response = self.model.generate_content(

            prompt,

            generation_config=GenerationConfig(temperature=self.temperature, **self.arguments),

            safety_settings=SAFETY_FILTER_CONFIG,

        ).text

        if parser_func:

            return parser_func(response)

        return response

 

    def call_parallel(

            self,

            prompts: List[str],

            parser_func: Optional[Callable[[str], str]] = None,

            timeout: int = 60,

            max_retries: int = 5,

    ) -> List[Optional[str]]:

        """Calls the Gemini model for multiple prompts in parallel."""

        results = [None] * len(prompts)

 

        def worker(index: int, prompt: str):

            retries = 0

            while retries <= max_retries:

                try:

                    return self.call(prompt, parser_func)

                except Exception as e:

                    print(f"Error for prompt {index}: {str(e)}")

                    retries += 1

                    if retries <= max_retries:

                        print(f"Retrying ({retries}/{max_retries}) for prompt {index}")

                        time.sleep(1)

                    else:

                        return f"Error after retries: {str(e)}"

 

        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:

            future_to_index = {

                executor.submit(worker, i, prompt): i

                for i, prompt in enumerate(prompts)

            }

            for future in as_completed(future_to_index, timeout=timeout):

                index = future_to_index[future]

                try:

                    results[index] = future.result()

                except Exception as e:

                    print(f"Unhandled error for prompt {index}: {e}")

                    results[index] = "Unhandled Error"

 

        for future in future_to_index:

            index = future_to_index[future]

            if not future.done():

                print(f"Timeout occurred for prompt {index}")

                results[index] = "Timeout"

 

        return results
