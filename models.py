import pathlib
from time import sleep
from typing import Generator, List, Tuple, Dict, Any

import google.generativeai as genai
from anthropic import Anthropic
from loguru import logger
from replicate import Client
from requests import Response, post, RequestException

import base64
import os
from abc import ABC, abstractmethod


class vLLM(ABC):
    """
    Base class for all vision language models.
    """

    def __init__(self, name_of_model: str, temp: float = 0.0, max_tokens: int = 1024) -> None:
        """
        Initializes the vision language model with the given parameters.

        Args:
            name_of_model (str): Name of the model used by the API (e.g., "gpt-4").
            temp (float): Temperature parameter for sampling from the model. Default is 0.0.
            max_tokens (int): Maximum number of tokens to generate. Default is 1024.

        Raises:
            ValueError: If `temp` is not within the range [0.0, 1.0] or if `max_tokens` is negative.
            TypeError: If `max_tokens` is not an integer.
        """
        if not (0.0 <= temp <= 1.0):
            raise ValueError("Temperature must be between 0.0 and 1.0.")

        if not isinstance(max_tokens, int):
            raise TypeError("max_tokens must be an integer.")

        if max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer.")

        self.name_of_model = name_of_model
        self.temp = temp
        self.max_tokens = max_tokens

    def __str__(self) -> str:
        """
        Returns a clean version of the model name, stripping any prefixes.

        Returns:
            str: The name of the model without any prefix.
        """
        return self.name_of_model.split("/")[-1]

    @abstractmethod
    def eval(self, prompt: str, path_to_image: str) -> str:
        """
        Evaluates the model with the provided prompt and image.

        Args:
            prompt (str): The query or question to ask the model regarding the image.
            path_to_image (str): Local filesystem path to the image file.

        Returns:
            str: The model's response to the prompt based on the image.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encodes the image at the specified path to a base64 string.

        Args:
            image_path (str): Local filesystem path to the image file.

        Returns:
            str: Base64 encoded string of the image content.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            IOError: If the image cannot be read.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file '{image_path}' not found.")

        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            raise IOError(f"An error occurred while reading the image file: {e}")


class GPTModel(vLLM):
    """
    Class for OpenAI's GPT models, such as:
        - gpt-4-turbo (vision)
        - gpt-4 (vision)
    """

    def __init__(self, name_of_model: str, temp: float, max_tokens: int) -> None:
        """
        Initializes the GPT model with the specified parameters.

        Args:
            name_of_model (str): The name of the GPT model (e.g., "gpt-4-turbo").
            temp (float): Temperature parameter for the model's sampling. Must be between 0.0 and 1.0.
            max_tokens (int): Maximum number of tokens to generate. Must be a non-negative integer.

        Raises:
            KeyError: If the 'OPENAI_API_KEY' environment variable is not set.
            ValueError: If `temp` is not in the range [0.0, 1.0] or `max_tokens` is negative.
            TypeError: If `max_tokens` is not an integer.
        """
        super().__init__(name_of_model, temp, max_tokens)
        try:
            self.api_key: str = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError("The 'OPENAI_API_KEY' environment variable is not set.")

        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def eval(
        self, prompt: str, path_to_image: str, max_retries: int = 50, delay: int = 1
    ) -> str:
        """
        Evaluates the GPT model with the given prompt and image.

        Args:
            prompt (str): The query or question to ask the model regarding the image.
            path_to_image (str): Local filesystem path to the image file.
            max_retries (int): Maximum number of retry attempts in case of request failure. Default is 50.
            delay (int): Delay in seconds between retry attempts. Default is 1.

        Returns:
            str: The model's response to the prompt.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            IOError: If the image cannot be read or encoded.
            RequestException: If the HTTP request fails after all retries.
        """
        attempts: int = 0
        while attempts < max_retries:
            try:
                base64_image: str = self.encode_image(path_to_image)
                payload: Dict[str, Any] = {
                    "model": self.name_of_model,
                    "temperature": self.temp,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": self.max_tokens,
                }
                response: Response = post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()  # Raise an error for bad status codes
                return response.json()["choices"][0]["message"]["content"]
            except (RequestException, IOError) as e:
                attempts += 1
                logger.info(f"Attempt {attempts} failed with error: {e}")
                sleep(delay)
                if attempts == max_retries:
                    logger.error(f"All {max_retries} attempts failed.")
                    raise RequestException(f"Failed after {max_retries} attempts") from e


class ReplicateModel(vLLM):
    """
    Class for Replicate's models
    """

    def __init__(self, name_of_model: str, temp: float, max_tokens: int) -> None:
        """
        Initializes the Replicate model with the specified parameters.

        Args:
            name_of_model (str): The name of the Replicate model (e.g., "lucataco/moondream2:72ccb656353c348c1385df54b237eeb7bfa874bf11486cf0b9473e691b662d31").
            temp (float): Temperature parameter for the model's sampling. Must be between 0.0 and 1.0.
            max_tokens (int): Maximum number of tokens to generate. Must be a non-negative integer.

        Raises:
            KeyError: If the 'REPLICATE_API_TOKEN' environment variable is not set.
            ValueError: If `temp` is not in the range [0.0, 1.0] or `max_tokens` is negative.
            TypeError: If `max_tokens` is not an integer.
        """
        super().__init__(name_of_model, temp, max_tokens)
        try:
            self.api_key: str = os.environ["REPLICATE_API_TOKEN"]
        except KeyError:
            raise KeyError("The 'REPLICATE_API_TOKEN' environment variable is not set.")

        self.replicate_client = Client(api_token=self.api_key)

    def eval(
            self, prompt: str, path_to_image: str, max_retries: int = 50, delay: int = 1
    ) -> str:
        """
        Evaluates the Replicate model with the given prompt and image.

        Args:
            prompt (str): The query or instruction to guide the model's output.
            path_to_image (str): Local filesystem path to the image file.
            max_retries (int): Maximum number of retry attempts in case of request failure. Default is 50.
            delay (int): Delay in seconds between retry attempts. Default is 1.

        Returns:
            str: The model's generated output as a string.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            IOError: If the image cannot be read.
            RequestException: If the request to Replicate fails after all retries.
        """
        attempts: int = 0
        while attempts < max_retries:
            try:
                with open(path_to_image, "rb") as image:
                    input_data = {
                        "image": image,
                        "prompt": prompt,
                        "temperature": self.temp,
                        "max_new_tokens": self.max_tokens,
                    }

                    output: Generator = self.replicate_client.run(
                        self.name_of_model, input=input_data
                    )
                    return "".join(output)
            except (RequestException, IOError) as e:
                attempts += 1
                logger.info(f"Attempt {attempts} failed with error: {e}")
                sleep(delay)
                if attempts == max_retries:
                    logger.error(f"All {max_retries} attempts failed.")
                    raise RequestException(f"Failed after {max_retries} attempts")


class AnthropicModel(vLLM):
    """
    Class for Anthropic's models.
    """

    def __init__(self, name_of_model: str, temp: float, max_tokens: int) -> None:
        """
        Initializes the Anthropic model with the specified parameters.

        Args:
            name_of_model (str): The name of the Anthropic model (e.g., "claude-3-5-sonnet-20240620").
            temp (float): Temperature parameter for the model's sampling. Must be between 0.0 and 1.0.
            max_tokens (int): Maximum number of tokens to generate. Must be a non-negative integer.

        Raises:
            KeyError: If the 'ANTHROPIC_API_KEY' environment variable is not set.
            ValueError: If `temp` is not in the range [0.0, 1.0] or `max_tokens` is negative.
            TypeError: If `max_tokens` is not an integer.
        """
        super().__init__(name_of_model, temp, max_tokens)
        try:
            self.api_key: str = os.environ["ANTHROPIC_API_KEY"]
        except KeyError:
            raise KeyError("The 'ANTHROPIC_API_KEY' environment variable is not set.")

        self.anthropic_client = Anthropic(api_key=self.api_key)

    def eval(
            self, prompt: str, path_to_image: str, max_retries: int = 50, delay: int = 1
    ) -> str:
        """
        Evaluates the Anthropic model with the given prompt and image.

        Args:
            prompt (str): The query or instruction to guide the model's output.
            path_to_image (str): Local filesystem path to the image file.
            max_retries (int): Maximum number of retry attempts in case of request failure. Default is 50.
            delay (int): Delay in seconds between retry attempts. Default is 1.

        Returns:
            str: The model's generated output as a string.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            IOError: If the image cannot be read.
            APIException: If the API request fails after all retries.
        """
        attempts: int = 0
        while attempts < max_retries:
            try:
                image: str = self.encode_image(path_to_image)
                message = self.anthropic_client.messages.create(
                    model=self.name_of_model,
                    temperature=self.temp,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                )
                return message.content[0].text
            except (RequestException, IOError) as e:
                attempts += 1
                logger.info(f"Attempt {attempts} failed with error: {e}")
                sleep(delay)
                if attempts == max_retries:
                    logger.error(f"All {max_retries} attempts failed.")
                    raise RequestException(f"Failed after {max_retries} attempts")


class GeminiModel(vLLM):
    """
    Class for Google's Gemini models.
    """

    def __init__(self, name_of_model: str, temp: float, max_tokens: int) -> None:
        """
        Initializes the Gemini model with the specified parameters.

        Args:
            name_of_model (str): The name of the Gemini model (e.g., "gemini-1.5-flash").
            temp (float): Temperature parameter for the model's sampling. Must be between 0.0 and 1.0.
            max_tokens (int): Maximum number of tokens to generate. Must be a non-negative integer.

        Raises:
            KeyError: If the 'GOOGLE_API_KEY' environment variable is not set.
            ValueError: If `temp` is not in the range [0.0, 1.0] or `max_tokens` is negative.
            TypeError: If `max_tokens` is not an integer.
        """
        super().__init__(name_of_model, temp, max_tokens)
        try:
            self.api_key: str = os.environ["GOOGLE_API_KEY"]
        except KeyError:
            raise KeyError("The 'GOOGLE_API_KEY' environment variable is not set.")

        self.gemini_client = genai
        self.gemini_client.configure(api_key=self.api_key)

    def eval(
            self, prompt: str, path_to_image: str, max_retries: int = 50, delay: int = 1
    ) -> str:
        """
        Evaluates the Gemini model with the given prompt and image.

        Args:
            prompt (str): The query or instruction to guide the model's output.
            path_to_image (str): Local filesystem path to the image file.
            max_retries (int): Maximum number of retry attempts in case of request failure. Default is 50.
            delay (int): Delay in seconds between retry attempts. Default is 1.

        Returns:
            str: The model's generated output as a string.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            IOError: If the image cannot be read.
            APIError: If the API request fails after all retries.
        """
        safety_settings: List[Dict[str, str]] = [
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        attempts: int = 0
        while attempts < max_retries:
            try:
                model = self.gemini_client.GenerativeModel(self.name_of_model)
                image = {
                    "mime_type": "image/jpeg",
                    "data": pathlib.Path(path_to_image).read_bytes(),
                }

                response = model.generate_content(
                    contents=[image, prompt],
                    safety_settings=safety_settings,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        stop_sequences=["x"],
                        max_output_tokens=self.max_tokens,
                        temperature=self.temp,
                    ),
                )
                return response.text
            except (RequestException, IOError) as e:
                attempts += 1
                logger.info(f"Attempt {attempts} failed with error: {e}")
                sleep(delay)
                if attempts == max_retries:
                    logger.error(f"All {max_retries} attempts failed.")
                    raise RequestException(f"Failed after {max_retries} attempts")

