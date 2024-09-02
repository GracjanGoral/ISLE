import pandas as pd


def prepare_isle_brick_dataset_question(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Isle-Brick dataset by creating questions with permuted answer options and adding the correct answer.

    Args:
        dataset (pd.DataFrame): The dataset to prepare, containing the following columns:
            - 'id_image': Identifier for the image associated with the question.
            - 'question': The question text.
            - 'gold_answer': The correct answer for the question.
            - 'partition': Type of image, for example partition 1 contains one mini figure with object, and partition 2 contains one mini figure with two objects.
            - 'type_of_dataset': Type of dataset, for example isle-bricks.
            - 'method': Method to prepare the dataset, for example 0-shot.
            - 'task': Task to prepare the dataset, for example perspective-taking.

    Returns:
        pd.DataFrame: A DataFrame containing the original image IDs, the original and permuted questions,
                      and the correct answer. The DataFrame has the following columns:
            - 'id_image': Identifier for the image.
            - 'question': The original question with options A and B.
            - 'permuted_question': The question with permuted options.
            - 'gold_answer': The correct answer (A or B).
            - 'partition': Type of image, for example partition 1 contains one mini figure with object, and partition 2 contains one mini figure with two objects.
            - 'type_of_dataset': Type of dataset, for example isle-bricks.
            - 'method': Method to prepare the dataset, for example 0-shot.
            - 'task': Task to prepare the dataset, for example perspective-taking.
    """

    # Initialize an empty DataFrame to store the results
    columns = [
        "id_image",
        "question",
        "permuted_question",
        "gold_answer",
        "partition",
        "type_of_dataset",
        "method",
        "task",
    ]
    prepared_dataset = pd.DataFrame(columns=columns)

    # Iterate through each row in the dataset
    for _, row in dataset.iterrows():
        id_image = row["id_image"]
        question = row["question"]
        gold_answer = row["gold_answer"]
        partition = row["partition"]
        type_of_dataset = row["type_of_dataset"]
        task = row["task"]

        # Construct the original and permuted prompts
        original_prompt = (
            f"{question} Respond solely with A or B.\nOptions:\nA. Yes\nB. No\nAnswer: "
        )
        permuted_prompt = (
            f"{question} Respond solely with A or B.\nOptions:\nA. No\nB. Yes\nAnswer: "
        )

        # Append the new data to the prepared dataset
        prepared_dataset = pd.concat(
            [
                prepared_dataset,
                pd.DataFrame(
                    [
                        {
                            "id_image": id_image,
                            "question": original_prompt,
                            "permuted_question": permuted_prompt,
                            "gold_answer": gold_answer,
                            "partition": partition,
                            "type_of_dataset": type_of_dataset,
                            "method": "0-shot",
                            "task": task,
                        }
                    ]
                ),
            ],
        )

    return prepared_dataset


def prepare_isle_brick_dataset_cot_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Isle-Brick dataset by creating questions with permuted answer options and adding the correct answer.
    This function to each question adds a prompt to analyze the image step by step which is known as Chain of Thought (COT) method.

     Args:
         dataset (pd.DataFrame): The dataset to prepare, containing the following columns:
            - 'id_image': Identifier for the image associated with the question.
            - 'question': The question text.
            - 'gold_answer': The correct answer for the question.
            - 'partition': Type of image, for example partition 1 contains one mini figure with object, and partition 2 contains one mini figure with two objects.
            - 'type_of_dataset': Type of dataset, for example isle-bricks.
            - 'method': Method to prepare the dataset, for example CoT.
            - 'task': Task to prepare the dataset, for example perspective-taking.

     Returns:
         pd.DataFrame: A DataFrame containing the original image IDs, the original and permuted questions,
                       and the correct answer. The DataFrame has the following columns:
            - 'id_image': Identifier for the image.
            - 'question': The original question with options A and B.
            - 'permuted_question': The question with permuted options.
            - 'gold_answer': The correct answer (A or B).
            - 'partition': Type of image, for example partition 1 contains one mini figure with object, and partition 2 contains one mini figure with two objects.
            - 'type_of_dataset': Type of dataset, for example isle-bricks.
            - 'method': Method to prepare the dataset, for example CoT.
            - 'task': Task to prepare the dataset, for example perspective-taking.
    """

    # Initialize an empty DataFrame to store the results
    columns = [
        "id_image",
        "question",
        "permuted_question",
        "gold_answer",
        "partition",
        "type_of_dataset",
        "method",
        "task",
    ]
    prepared_dataset = pd.DataFrame(columns=columns)

    # Iterate through each row in the dataset
    for _, row in dataset.iterrows():
        id_image = row["id_image"]
        question = row["question"]
        gold_answer = row["gold_answer"]
        partition = row["partition"]
        type_of_dataset = row["type_of_dataset"]
        task = row["task"]

        # Construct the original and permuted prompts
        original_prompt = f"{question} First, analyze the image step by step, then provide your answer by selecting either option A or B.\nOptions:\nA. Yes\nB. No\nAnswer: "
        permuted_prompt = f"{question} First, analyze the image step by step, then provide your answer by selecting either option A or B.\nOptions:\nA. No\nB. Yes\nAnswer: "

        # Append the new data to the prepared dataset
        prepared_dataset = pd.concat(
            [
                prepared_dataset,
                pd.DataFrame(
                    [
                        {
                            "id_image": id_image,
                            "question": original_prompt,
                            "permuted_question": permuted_prompt,
                            "gold_answer": gold_answer,
                            "partition": partition,
                            "type_of_dataset": type_of_dataset,
                            "method": "CoT",
                            "task": task,
                        }
                    ]
                ),
            ],
        )

    return prepared_dataset


def prepare_isle_dots_dataset_question(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the Isle-DOts dataset by creating questions with permuted answer options and adding the correct answer.
    New options are created by adding 1 to the correct answer (gold_answer).

    Args:
        dataset (pd.DataFrame): The dataset to prepare, containing the following columns:
            - 'id_image': Identifier for the image associated with the question.
            - 'question': The question text.
            - 'gold_answer': The correct answer for the question.
            - 'partition': Type of image, for instance, Partition 1 includes images where there is a single figure displayed on a wall, accompanied by a person who is turned in a specific direction. On the other hand, Partition 2 consists of images featuring two figures on walls, along with a person who is similarly turned in a particular direction
            - 'type_of_dataset': Type of dataset, for example isle-dots.
            - 'method': Method to prepare the dataset, for example 0-shot.
            - 'task': Task to prepare the dataset, for example perspective-taking.

    Returns:
        pd.DataFrame: A DataFrame containing the original image IDs, the original and permuted questions,
                      and the correct answer. The DataFrame has the following columns:
            - 'id_image': Identifier for the image.
            - 'question': The original question with options A and B.
            - 'permuted_question': The question with permuted options.
            - 'gold_answer': The correct answer (A or B).
            - 'partition': Type of image, for instance, Partition 1 includes images where there is a single figure displayed on a wall, accompanied by a person who is turned in a specific direction. On the other hand, Partition 2 consists of images featuring two figures on walls, along with a person who is similarly turned in a particular direction
            - 'type_of_dataset': Type of dataset, for example isle-dots.
            - 'method': Method to prepare the dataset, for example 0-shot.
            - 'task': Task to prepare the dataset, for example perspective-taking.
    """

    columns = [
        "id_image",
        "question",
        "permuted_question",
        "gold_answer",
        "partition",
        "type_of_dataset",
        "method",
        "task",
    ]
    prepared_dataset = pd.DataFrame(columns=columns)

    # Iterate through each row in the dataset
    for _, row in dataset.iterrows():
        id_image = row["id_image"]
        question = row["question"]
        gold_answer = row["gold_answer"]
        partition = row["partition"]
        type_of_dataset = row["type_of_dataset"]
        task = row["task"]
        new_option_answer = str(int(gold_answer) + 1)

        # Construct the original and permuted prompts
        original_prompt = f"{question} Respond solely with A or B.\nOptions:\nA. {gold_answer}\nB. {new_option_answer}\nAnswer: "
        permuted_prompt = f"{question} Respond solely with A or B.\nOptions:\nA. {new_option_answer}\nB. {gold_answer}\nAnswer: "

        # Append the new data to the prepared dataset
        prepared_dataset = pd.concat(
            [
                prepared_dataset,
                pd.DataFrame(
                    [
                        {
                            "id_image": id_image,
                            "question": original_prompt,
                            "permuted_question": permuted_prompt,
                            "gold_answer": gold_answer,
                            "partition": partition,
                            "type_of_dataset": type_of_dataset,
                            "method": "0-shot",
                            "task": task,
                        }
                    ]
                ),
            ],
        )

    return prepared_dataset


def prepare_isle_dots_dataset_cot_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """

    Prepare the Isle-DOts dataset by creating questions with permuted answer options and adding the correct answer.
    New options are created by adding 1 to the correct answer (gold_answer).
    This function to each question adds a prompt to analyze the image step by step which is known as Chain of Thought (COT) method.

    Args:
        dataset (pd.DataFrame): The dataset to prepare, containing the following columns:
            - 'id_image': Identifier for the image associated with the question.
            - 'question': The question text.
            - 'gold_answer': The correct answer for the question.
            - 'partition': Type of image, for instance, Partition 1 includes images where there is a single figure displayed on a wall, accompanied by a person who is turned in a specific direction. On the other hand, Partition 2 consists of images featuring two figures on walls, along with a person who is similarly turned in a particular direction
            - 'type_of_dataset': Type of dataset, for example isle-dots.
            - 'method': Method to prepare the dataset, for example CoT.
            - 'task': Task to prepare the dataset, for example perspective-taking.

    Returns:
        pd.DataFrame: A DataFrame containing the original image IDs, the original and permuted questions,
                      and the correct answer. The DataFrame has the following columns:
            - 'id_image': Identifier for the image.
            - 'question': The original question with options A and B.
            - 'permuted_question': The question with permuted options.
            - 'gold_answer': The correct answer (A or B).
            - 'partition': Type of image, for instance, Partition 1 includes images where there is a single figure displayed on a wall, accompanied by a person who is turned in a specific direction. On the other hand, Partition 2 consists of images featuring two figures on walls, along with a person who is similarly turned in a particular direction
            - 'type_of_dataset': Type of dataset, for example isle-dots.
            - 'method': Method to prepare the dataset, for example CoT.
            - 'task': Task to prepare the dataset, for example perspective-taking.

    """

    columns = [
        "id_image",
        "question",
        "permuted_question",
        "gold_answer",
        "partition",
        "type_of_dataset",
        "method",
        "task",
    ]
    prepared_dataset = pd.DataFrame(columns=columns)

    # Iterate through each row in the dataset
    for _, row in dataset.iterrows():
        id_image = row["id_image"]
        question = row["question"]
        gold_answer = row["good_answer"]
        partition = row["partition"]
        type_of_dataset = row["type_of_dataset"]
        task = row["task"]
        new_option_answer = str(int(gold_answer) + 1)

        # Construct the original and permuted prompts
        original_prompt = f"{question} First, analyze the image step by step, then provide your answer by selecting either option A or B.\nOptions:\nA. {gold_answer}\nB. {new_option_answer}\nAnswer: "
        permuted_prompt = f"{question} First, analyze the image step by step, then provide your answer by selecting either option A or B.\nOptions:\nA. {new_option_answer}\nB. {gold_answer}\nAnswer: "

        # Append the new data to the prepared dataset
        prepared_dataset = pd.concat(
            [
                prepared_dataset,
                pd.DataFrame(
                    [
                        {
                            "id_image": id_image,
                            "question": original_prompt,
                            "permuted_question": permuted_prompt,
                            "gold_answer": gold_answer,
                            "partition": partition,
                            "type_of_dataset": type_of_dataset,
                            "method": "CoT",
                            "task": task,
                        }
                    ]
                ),
            ],
        )
    return prepared_dataset
