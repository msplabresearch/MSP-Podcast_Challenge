# MSP-Podcast Emotion Challenge - Baselines Training and Evaluation

This repository contains scripts to train and evaluate baseline models on various tasks including categorical emotion, multi-task emotional attributes, and single-task emotional attributes for arousal, dominance, and valence.

Link to the challenge: [Odyssey 2024 - Emotion Recognition Challenge](https://www.odyssey2024.org/emotion-recognition-challenge)

Link to the [submission website](https://lab-msp.com/MSP-Podcast_Competition/leaderboard.php)

Refer to the links above to sign-up for the challenge, rules, submission deadlines, and file formatting instructions.
## Environment Setup

To replicate the environment necessary to run the code, you have two options:

### Using Conda

1. Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed.
2. Create a conda environment using the `spec-file.txt` by running:
    conda create --name baseline_env --file spec-file.txt
3. Activate the environment:
    conda activate baseline_env
4. Make sure to install the transformers library as it is essential for the code to run:
    pip install transformers



### Using pip

1. Alternatively, you can use `requirements.txt` to install the necessary packages in a virtual environment:
    python -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt
2. Make sure to install the transformers library as it is essential for the code to run:
    pip install transformers


## Configuration

Before running the training or evaluation scripts, update the `config.json` file with the paths to your local audio folder and label CSV file.

## Training and Evaluation

To train or evaluate the models, use the provided shell scripts. Here's how to use each script:

- `bash run_cat.sh`: Trains or evaluates the categorical emotion recognition baseline.
- `bash run_dim.sh`: Trains or evaluates the multi-task emotional attributes baseline.
- `bash run_arousal.sh`: Trains or evaluates the single-task emotional attribute baseline for arousal.
- `bash run_dominance.sh`: Trains or evaluates the single-task emotional attribute baseline for dominance.
- `bash run_valence.sh`: Trains or evaluates the single-task emotional attribute baseline for valence.


### Models

The models are to be saved in the `model` folder. If you are evaluating the pretrained models, please download the models using the script provided in `model` folder. 
  ```
  $ bash download_models.sh <categorical|arousal|dominance|valence|multitask|all>
  ```
- Example 1: `bash download_models.sh all` to download all the models.
- Example 2: `bash download_models.sh arousal valence` to download arousal and valence models.

If you wish to manually download a pre-trained model. Please visit this [website](https://lab-msp.com/MODELS/Odyssey_Baselines/) and download the desired model and place them in the `model` folder. 

Pre-trained models file descriptions:
- "weight_cat_ser.zip" --> Categorical emotion recognition baseline.
- "dim_aro_ser.zip" --> Arousal single-task emotional attribute baseline.
- "dim_dom_ser.zip" --> Dominance single-task emotional attribute baseline.
- "dim_val_ser.zip" --> Valence single-task emotional attribute baseline.
- "dim_ser.zip" --> Multi-task emotional attributes baseline.
    
### Evaluation Only

If you are only evaluating a model and do not wish to train it, comment out the lines related to `train_****_.py` in the respective `.sh` file.

## Issues

If you encounter any issues while setting up, training, or evaluating the models, please open an issue on this repository with a detailed description of the problem.
