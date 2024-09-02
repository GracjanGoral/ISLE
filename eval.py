from answer_mapping import MapAnswerOpenAI, Mapper
from models import vLLM
import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def eval_perspective_taking(
    model: vLLM, dataset: pd.DataFrame, mapper: Mapper,
) -> pd.DataFrame:
    """
    Evaluates vision-language models on the task of perspective taking/object detection.

    Args:
        model: An instance of the model to be evaluated. Supported companies: OpenAI, Anthropic, Gemini (Google), Replicate.
        dataset: The dataset used for evaluation.
        mapper: A method to map the model's answer to one of the options. You can use OpenAI models like GPT-4, or turn it off and map manually (in this case, the values will be None).

    Returns:
        DataFrame: A DataFrame containing the original dataset columns and additional columns, including:
                   - model answer: The response from the evaluated model.
                   - mapped answer: The mapped model answer to one of the options.
                   - Other relevant metadata.
    """

    try:
        results_folder = "evaluation_results/"
        os.makedirs(results_folder, exist_ok=True)
        logger.info(f"Results folder '{results_folder}' created or already exists.")
    except Exception as e:
        logger.error(f"Failed to create results folder: {str(e)}")
        raise


    name_of_evaluated_model: str = str(model)
    name_of_dataset: str = dataset["type_of_dataset"].unique()[0]
    method: str = dataset["method"].unique()[0]
    task: str = dataset["task"].unique()[0]
    logger.info(
        f"Starting evaluation: Model={name_of_evaluated_model}, Dataset={name_of_dataset}, Task={task}, Method={method}"
    )

    results = pd.DataFrame(
        columns=[
            "path_to_id_image",
            "question",
            "model_answer",
            "mapped_answer",
            "gold_answer",
            "permuted",
            "type_of_dataset",
            "method",
            "task",
            "partition",
            "model_id",
        ]
    )

    for index, row in dataset.iterrows():
        try:
            id_image = row["id_image"]
            question = row["question"]
            permuted_question = row["permuted_question"]
            gold_answer = row["gold_answer"]
            partition = row["partition"]

            for permuted, prompt in enumerate([question, permuted_question]):
                logger.info(
                    f"Evaluating image ID {id_image} with {'original' if permuted == 0 else 'permuted'} question."
                )
                model_output: str = model.eval(prompt=prompt, path_to_image=id_image)
                mapped_answer: str = mapper.map_answer(
                    context_question=prompt, model_answer=model_output
                )

                result_row = {
                    "path_to_id_image": id_image,
                    "question": prompt,
                    "model_answer": model_output,
                    "mapped_answer": mapped_answer,
                    "gold_answer": gold_answer,
                    "permuted": permuted == 1,
                    "type_of_dataset": row["type_of_dataset"],
                    "method": method,
                    "task": task,
                    "partition": partition,
                    "model_id": name_of_evaluated_model,
                }
                results = pd.concat(
                    [results, pd.DataFrame([result_row])], ignore_index=True
                )
                logger.info(f"Appended results for image ID {id_image}.")

        except Exception as e:
            logger.error(f"Error processing row {index}: {str(e)}")
            continue

    results_file = f"{results_folder}/{name_of_dataset}_{task}_{method}_{name_of_evaluated_model}.csv"
    results.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}.")

    return results
