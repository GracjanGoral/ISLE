import argparse
import logging

import pandas as pd
from eval import eval_perspective_taking
from mapper_selector import mapper_model_selector
from model_selector import model_selector
from tools_dataset import (
    prepare_isle_brick_dataset_question,
    prepare_isle_brick_dataset_cot_dataset,
    prepare_isle_dots_dataset_question,
    prepare_isle_dots_dataset_cot_dataset,
)


def prepare_data_with_path(
    path_to_data: str, path_to_folder_with_images: str, method: str, dataset_type: str
) -> pd.DataFrame:
    """
    Prepares the dataset based on the given method and dataset type.

    Args:
        path_to_data (str): Path to the CSV data file.
        path_to_folder_with_images (str): Path to the folder containing images.
        method (str): The method to use ('0-shot' or 'CoT').
        dataset_type (str): The type of dataset ('isle-brick' or 'isle-dots').

    Returns:
        pd.DataFrame: Processed DataFrame.

    Raises:
        ValueError: If the method or dataset type is unsupported.
    """
    logging.info("Loading data from %s", path_to_data)
    data = pd.read_csv(path_to_data)

    assert data["type_of_dataset"].iloc[0] in ["isle-brick", "isle-dots"], f"Unsupported dataset type: {data['type_of_dataset'].iloc[0]}. Currently supported dataset types: isle-brick, isle-dots."
    assert method in ["0-shot", "CoT"], f"Unsupported method: {method}. Currently supported methods: 0-shot, CoT."

    if dataset_type == "isle-brick":
        if method == "0-shot":
            data = prepare_isle_brick_dataset_question(data)
        elif method == "CoT":
            data = prepare_isle_brick_dataset_cot_dataset(data)
        else:
            raise ValueError(f"Unsupported method: {method}")
    elif dataset_type == "isle-dots":
        if method == "0-shot":
            data = prepare_isle_dots_dataset_question(data)
        elif method == "CoT":
            data = prepare_isle_dots_dataset_cot_dataset(data)
        else:
            raise ValueError(f"Unsupported method: {method}")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    data["id_image"] = data["id_image"].apply(
        lambda x: f"{path_to_folder_with_images}/{x}"
    )

    return data


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare data and evaluate a model.")

    parser.add_argument("--data_path", required=True, help="Path to the CSV data file.")
    parser.add_argument(
        "--image_base_path", required=True, help="Path to the folder containing images."
    )
    parser.add_argument(
        "--model_name", default="gpt-4", help="Name of the model to use."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature setting for the model."
    )
    parser.add_argument(
        "--method", choices=["0-shot", "CoT"], default="CoT", help="Method to use."
    )
    parser.add_argument(
        "--dataset_type",
        choices=["isle-brick", "isle-dots"],
        default="isle-brick",
        help="Type of dataset."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=16, help="Max new tokens for the model."
    )
    parser.add_argument(
        "--mapper_selector",
        help="Name of the mapper model to use."
    )
    parser.add_argument(
        "--mapper_temperature",
        type=float,
        default=0.0,
        help="Temperature setting for the mapper model."
    )
    parser.add_argument(
        "--mapper_max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens for the mapper model."
    )

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_arguments()

    data = prepare_data_with_path(
        path_to_data=args.data_path,
        path_to_folder_with_images=args.image_base_path,
        method=args.method,
        dataset_type=args.dataset_type,
    )

    model = model_selector(
        name_of_model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens
    )


    mapper = mapper_model_selector(
        name_of_model=args.mapper_selector,
        temperature=args.mapper_temperature,
        max_tokens=args.mapper_max_new_tokens
    )

    eval_perspective_taking(model, data, mapper)