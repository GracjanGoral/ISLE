import json
import logging
import os
from abc import ABC, abstractmethod
from openai import OpenAI

# Ensure logger is set up
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Mapper(ABC):
    """
    An abstract class that defines the interface for mapping model answers to predefined options.
    """

    @staticmethod
    def create_json_payload_ab(input_string: str, model_answer: str) -> str:
        """
        Creates a JSON payload for mapping a model answer to 'Yes' or 'No' based on the provided context and options.

        Args:
            input_string (str): The input string containing the context and options.
            model_answer (str): The answer generated by the model to be mapped.

        Returns:
            str: A JSON string representing the payload.

        Raises:
            ValueError: If there is an issue processing the input string.
        """
        try:
            context, options_block = input_string.split("\nOptions:\n")
            options_block = options_block.replace("Answer: ", "").strip()

            options_dict = {
                opt.split(". ")[0]: opt.split(". ")[1]
                for opt in options_block.split("\n")
            }

            json_payload = {
                "INSTRUCTION": "Map the MODEL_ANSWER to 'Yes' or 'No' based on the CONTEXT and the provided OPTIONS.",
                "DETAILS": {
                    "CONTEXT": context,
                    "OPTIONS": options_dict,
                    "RESPONSE_FORMAT": "The response must be 'Yes', 'No', or 'Unknown' based on the matching of the MODEL_ANSWER to the OPTIONS provided.",
                },
                "MODEL_ANSWER": model_answer,
            }

            return json.dumps(json_payload, indent=4)
        except Exception as e:
            logger.error(f"Error creating JSON payload for AB mapping: {e}")
            raise ValueError(f"Error processing input: {str(e)}") from e

    @staticmethod
    def create_json_payload_number(input_string: str, model_answer: str) -> str:
        """
        Creates a JSON payload for mapping a model answer to one of the provided options.

        Args:
            input_string (str): The input string containing the context and options.
            model_answer (str): The answer generated by the model to be mapped.

        Returns:
            str: A JSON string representing the payload.

        Raises:
            ValueError: If there is an issue processing the input string.
        """
        try:
            context, options_block = input_string.split("\nOptions:\n")
            options_block = options_block.replace("Answer: ", "").strip()

            options_dict = {
                option.split(". ")[0]: option.split(". ")[1]
                for option in options_block.split("\n")
            }

            valid_responses = ", ".join(
                [f"'{value}'" for value in options_dict.values()]
            )

            json_payload = {
                "INSTRUCTION": f"Map the MODEL_ANSWER to one of the values {valid_responses} based on the CONTEXT and the provided OPTIONS. Map to 'Unknown' if no match is found.",
                "DETAILS": {
                    "CONTEXT": context,
                    "OPTIONS": options_dict,
                    "RESPONSE_FORMAT": f"The response must be one of the values {valid_responses} or 'Unknown'.",
                },
                "MODEL_ANSWER": model_answer,
            }

            return json.dumps(json_payload, indent=4)
        except Exception as e:
            logger.error(f"Error creating JSON payload for number mapping: {e}")
            raise ValueError(f"Error processing input: {str(e)}") from e

    @abstractmethod
    def map_answer(self, context_question: str, model_answer: str) -> str:
        """
        Maps the model's answer to the correct option based on the context and returns the mapped result.

        Args:
            context_question (str): The input context and options as a string.
            model_answer (str): The answer generated by the model to be mapped.

        Returns:
            str: The final mapped answer based on the context and options provided.
        """
        raise NotImplementedError("This method must be implemented.")


class MapAnswerOpenAI(Mapper):
    """
    A class that handles mapping model answers to predefined options using OpenAI's models.
    """

    def __init__(
        self, name_of_model: str, temp: float = 0.0, max_tokens: int = 16
    ) -> None:
        """
        Initializes the MapAnswerOpenAI instance with the specified model.

        Args:
            name_of_model (str): The name of the OpenAI model to use.

        Raises:
            KeyError: If the 'OPENAI_API_KEY' environment variable is not set.
        """
        try:
            self.api_key: str = os.environ["OPENAI_API_KEY"]
        except KeyError:
            raise KeyError("The 'OPENAI_API_KEY' environment variable is not set.")

        try:
            self.open_ai = OpenAI(api_key=self.api_key)
            self.model_name = name_of_model
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise RuntimeError("Failed to initialize OpenAI client") from e

        self.temp = temp
        self.max_tokens = max_tokens

    def map_answer(self, context_question: str, model_answer: str) -> str:
        """
        Maps the model's answer to the correct option based on the context and returns the mapped result.

        Args:
            context_question (str): The input context and options as a string.
            model_answer (str): The answer generated by the model to be mapped.

        Returns:
            str: The final mapped answer based on the context and options provided.

        Raises:
            RuntimeError: If the API call to OpenAI fails.
            ValueError: If there is an issue creating the JSON payload.
        """
        try:
            json_result = self.create_json_payload_number(
                context_question, model_answer
            )
            response = self.open_ai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": json_result}],
                temperature=self.temp,
                max_tokens=self.max_tokens,
            )
            return response.model_dump()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise RuntimeError("Failed to map answer with OpenAI API") from e


class MapAnswerToNone(Mapper):
    """
    A class that always maps the model's answer to 'None'.
    """

    def map_answer(self, context_question: str, model_answer: str) -> str:
        """
        Maps the model's answer to 'None' regardless of the context.

        Args:
            context_question (str): The input context and options as a string.
            model_answer (str): The answer generated by the model to be mapped.

        Returns:
            str: 'None'
        """
        return "None"
