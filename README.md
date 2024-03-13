# MSP-Podcast Emotion Challenge - Baselines Training and Evaluation

This repository contains scripts to train and evaluate baseline models on various tasks including categorical emotion, multi-task emotional attributes, and single-task emotional attributes for arousal, dominance, and valence.

Link to the baseline experiments: [Baseline_Model.pdf](Baseline_Model.pdf)

Link to the challenge: [Odyssey 2024 - Emotion Recognition Challenge](https://www.odyssey2024.org/emotion-recognition-challenge)

Link to the [submission website](https://lab-msp.com/MSP-Podcast_Competition/leaderboard.php)

Refer to the links above to sign-up for the challenge, rules, submission deadlines, and file formatting instructions.
## Environment Setup

Python version = 3.9.7

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

Before running the training or evaluation scripts, check intructions below and update the `config.json` file with the paths to your local audio folder and label CSV file.

### Categorical Emotion Recognition Model

Before running training or evaluation of categorical emotion recognition model. Please execute the script `process_labels_for_categorical.py` to properly format the provided `labels_consensus.csv` file for categorical emotion recognition. Then place the path of the processed .csv file in the `config.json` file to run this configuration.

### Attributes Emotion Recognition Model

The original `labels_consensus.csv` file provided with the dataset can be used as-is for attributes emotion recognition. Please place the path to the `labels_consensu.csv` file in the `config.json` file to run this configuration.

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

### HuggingFace

If you are only interested in using the pretained models for prediction or feature extraction, we have made the models available on HuggingFace.

  #### Models on HuggingFace
  - [x] [Categorical model](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Categorical)
  - [x] [Multi-Task attribute model](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes)
  - [ ] Valence model
  - [ ] Dominance model
  - [ ] Arousal model

### Evaluation Only

If you are only evaluating a model and do not wish to train it, comment out the lines related to `train_****_.py` in the respective `.sh` file.

### Evaluation and saving results for emotional attributes prediction

A custom executable sample file for evaluation and results saving has been provided `eval_dim_ser_test3.py`. To execute, just download or train the multi-task emotional attributes baseline as a sample. Then execute `bash run_dim.sh` (NOTE: if you will not be training the entire model again. Please, comment out the training lines in `run_dim.sh` before evaluation). The saved results will be saved in the correct `.csv` format and it will be located in a `results` folder created inside your model path location.

## Issues

If you encounter any issues while setting up, training, or evaluating the models, please open an issue on this repository with a detailed description of the problem.

---------------------------
To cite this repository in your works, use the following BibTeX entry:

```
@InProceedings{Goncalves_2024,
            author={L. Goncalves and A. N. Salman and A. {Reddy Naini} and L. Moro-Velazquez and T. Thebaud and L. {Paola Garcia} and N. Dehak and B. Sisman and C. Busso},
            title={Odyssey2024 - Speech Emotion Recognition Challenge: Dataset, Baseline Framework, and Results},
            booktitle={Odyssey 2024: The Speaker and Language Recognition Workshop)},
            volume={To appear},
            year={2024},
            month={June},
            address =  {Quebec, Canada},
}
```

