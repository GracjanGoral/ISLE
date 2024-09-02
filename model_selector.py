from typing import Dict, List

from tabulate import tabulate

from models import vLLM, GPTModel, AnthropicModel, GeminiModel, ReplicateModel
from supported_models import SUPPORTED_MODELS


def supported_models(models: Dict[str, List[str]], model_name: str = None):
    if not models:
        raise ValueError(
            "No models provided. Please check the company's website for the latest supported models."
        )

    table_data = []
    for company, model_list in models.items():
        for model in model_list:
            table_data.append([company, model])

    print(tabulate(table_data, headers=["Company", "Model"], tablefmt="pretty"))

    if model_name:
        model_found = any(model_name in model_list for model_list in models.values())
        if not model_found:
            raise ValueError(
                f"The model '{model_name}' is not found. Please check the company's website for the latest supported models."
            )
        else:
            print(f"Model '{model_name}' is supported.")


def model_selector(
    name_of_model: str, temperature: float = 0.0, max_tokens: int = 1024
) -> vLLM:
    supported_models(SUPPORTED_MODELS, name_of_model)
    if name_of_model in SUPPORTED_MODELS["OpenAI"]:
        return GPTModel(
            name_of_model=name_of_model, temp=temperature, max_tokens=max_tokens
        )
    elif name_of_model in SUPPORTED_MODELS["Anthropic"]:
        return AnthropicModel(
            name_of_model=name_of_model, temp=temperature, max_tokens=max_tokens
        )
    elif name_of_model in SUPPORTED_MODELS["Google"]:
        return GeminiModel(
            name_of_model=name_of_model, temp=temperature, max_tokens=max_tokens
        )
    elif name_of_model in SUPPORTED_MODELS["Replicate"]:
        return ReplicateModel(
            name_of_model=name_of_model, temp=temperature, max_tokens=max_tokens
        )
    else:
        raise ValueError(
            f"Model '{name_of_model}' is not supported. Please check the company's website for the latest supported models."
        )
