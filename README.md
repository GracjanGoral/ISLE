<p align="center">
  <img src="logo.png" alt="Relevant Description" width="160" height="150">
</p>

<p align="center" style="font-size: 24px;">
  <strong>Seeing Through Their Eyes:<br>Evaluating Visual Perspective Taking in Vision Language Models</strong>
</p>


<p align="center">
  <a href="https://sites.google.com/view/perspective-taking/strona-g%C5%82%C3%B3wna" target="_blank" style="text-decoration: none;">
    <button style="background-color: #1E90FF; border: none; color: white; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin-top: 20px; cursor: pointer; border-radius: 12px;">
      Visit Project Website
    </button>
  </a>
</p>


## Project Description
This repository is dedicated to evaluating visual perspective taking in vision-language models. It hosts the code used in the paper titled **"Seeing Through Their Eyes: Evaluating Visual Perspective Taking in Vision Language Models"**, which can be accessed [link].
## Project Setup

### 1. Create a New Python Project

To start, set up a Python environment by following these steps:

```bash
# Install Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
### 2. Clone the repository and navigate into the project directory
```bash
# Clone the repository using the command below
git clone [link]

# Navigate into the project directory
cd ISLE
```
### 3. Install the Required Libraries
```bash
# Install the required libraries
pip install -r requirements.txt
```
### 4. API Token Configuration
Configure API tokens to allow your scripts to communicate with external model APIs.
This setup is essential for integrating services such as OpenAI, Google (Gemini), Anthropic, and Replicate.

```bash
# Example configuration for multiple API tokens
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
export REPLICATE_API_TOKEN="your_replicate_api_token_here"
```
**Note:**
If you want to avoid manually mapping model answers, you can utilize another language model like GPT-4.
To do so, you will need to provide an OpenAI token (currently, only GPT models are supported for mapping).
Alternatively, you can disable this option by providing `"manually"` to the `--mapper_selector` option (see **Running Evaluation** section). This will result in all answers being mapped to `None`.

## Dataset Description

The dataset structure is divided into two main components:

### 1. Images Folder
This folder contains a series of sequentially numbered images, ranging from `1.png` to `n.png`, where `n` is the size of the dataset. For a visual example of the images, see the folder named `example_of_images`.

### 2. CSV File
The associated CSV file contains the following fields:
- **`id_image`**: The number corresponding to each image, formatted as `"1.png"`.
- **`question`**: A question posed for each image. Example: `"Assuming the snail Lego minifigure has the ability to see, can it see the flower in the picture?"`
- **`gold_answer`**: The correct answer to the question.
- **`partition`**: A classification of images, e.g., the first partition contains images with only one figure and one object, while the second partition includes images with one figure and two objects.
- **`type_of_dataset`**: Currently supported types are `"isle-brick"` and `"isle-dots"`. More details can be found in the referenced article [link].
- **`task`**: The tasks available in this dataset include object detection (baseline) and perspective taking.

### 3. Task Flexibility
In addition to the predefined tasks, this construction of dataset allows the definition of any task that requires selecting options such as "Yes" or "No" (referred to as `"isle-brick"`), or selecting a number (referred to as `"isle-dots"`).
This means you can define another task with the dataset name, either `"isle-brick"` or `"isle-dots"`, which requires selecting either a yes/no option or a numerical value.
## Running Evaluation

Execute your script with the necessary parameters:

```bash
python3 run.py \
    --data_path path/to/data.csv \
    --image_base_path path/to/images \
    --model_name gpt-4o \
    --temperature 0.0 \
    --method 0-shot \
    --dataset_type isle-brick \
    --mapper_selector gpt-4 \
    --mapper_temperature 0.0 \
    --mapper_max_new_tokens 16
```
### Parameters Description

- `--data_path`: Path to the data file which contains the questions, type of dataset, and correct answers. See the file `example_of_data_perspective_taking.csv` for more details.
- `--image_base_path`: Path to the folder where the images are stored. Ensure you provide the correct path to both the images and the path where questions are stored. You can use `example_of_images` as an example.
- `--model_name`: Name of the model which will be used to generate answers. This is the model we use for testing perspective taking.
- `--temperature`: Controls the randomness of the model's responses. The default setting is 0.0.
- `--method`: The approach used to generate answers. Currently, we support 0-shot and Chain of Thought (CoT) methods.
- `--dataset_type`: "isle-brick" or "isle-dots".
- `--mapper_selector`: Method for mapping the model's answer to options. Currently, only OpenAI models are supported. You can disable this option by setting it to "manually," in which case each answer will be mapped to "None."
- `--mapper_temperature`: Temperature setting for the model used to map the answer. This parameter is ignored if the "manually" option is selected.
- `--mapper_max_new_tokens`: Maximum number of tokens returned by the model used to map the answer. This parameter is ignored if the "manually" option is selected.

After running the `run.py` script, you should find a folder named `evaluation_results`, containing files in the following format: `{results_folder}/{name_of_dataset}_{task}_{method}_{name_of_evaluated_model}.csv`.

This file includes detailed results from the evaluation, including:
- **`path_to_id_image`**: Path to the image.
- **`prompt`**: Prompt used to question the model.
- **`model_answer`**: Answer generated by the model.
- **`mapped_answer`**: Answer mapped to the options.
- **`gold_answer`**: Correct answer.
- **`permuted`**: Whether the answer was permuted.

### Adding a New Model Mapper

To add a new model mapper, follow these steps:

1. **Create a New Mapper Class**:  
   Define a class that inherits from the `Mapper` abstract base class. This class will serve as the blueprint for your model mapper.

    ```python
    class Mapper(ABC):
        """
        An abstract class that defines the interface for mapping model answers to predefined options.
        """ 
    ```

2. **Override the `map_answer` Method**:  
   Implement the `map_answer` method to map the model's answer to the correct option based on the context. This method must be overridden in your subclass.

    ```python
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
    ```

4. **Integrate the New Mapper**:  
   Add your newly created model mapper class to the `supported_models.py` file and update the `mapper_selector.py` to include your new mapper. 


## Future Improvements:
   - Adding batch inference to evaluate multiple models simultaneously.
   - Adding tools to visualize results.
