\# DeepSense 6G Capstone Project



This repository contains my capstone project based on the DeepSense 6G dataset.

The project focuses on beam prediction in 6G vehicle-to-infrastructure (V2I) communication scenarios using deep learning and multimodal sensor data.



The main experiment of this project is based on DeepSense 6G Scenario 32. In this scenario, multimodal/fusion-based beam prediction is studied. In addition to the dashboard and trained model, Scenario 32 training-related files were also added to the project.



Scenario 6 was also used during the project as an additional comparison experiment. However, Scenario 6 model files were not uploaded to this repository because the trained model weights were too large for standard GitHub upload limits. Therefore, Scenario 6 is mentioned as a comparison setup, but the repository mainly focuses on Scenario 32.



\## Project Overview



Beam prediction is an important task in 6G and mmWave communication systems. Since mmWave communication depends on highly directional beams, selecting the correct beam is necessary for maintaining reliable and efficient communication.



In this project, learning-based methods are used to predict the best beam index by using DeepSense 6G sensor data. The project mainly focuses on multimodal/fusion-based beam prediction.



The repository includes:



```text

\- Scenario 32 multimodal/fusion-based beam prediction

\- Scenario 32 training files

\- Trained multimodal beam prediction model

\- Dashboard for visualization and analysis

\- Notes about Scenario 6 RGB-based comparison experiment

```



\## Scenario Information



\### Scenario 32 - Main Experiment



Scenario 32 is the main scenario used in this repository.

It is used for multimodal/fusion-based beam prediction.



The Scenario 32 part focuses on:



```text

\- Multimodal beam prediction

\- Sensor fusion

\- Training and evaluation of deep learning models

\- Best beam index prediction

\- Dashboard-based result analysis

```



The trained model file included in this repository is:



```text

best\_multimodal\_beam\_model.pth

```



This model represents the trained multimodal beam prediction approach for the Scenario 32 experiment.



\### Scenario 32 Training



In addition to the trained model and dashboard, Scenario 32 training-related files were added to the project. These files are used to train or reproduce the Scenario 32 beam prediction model.



Users who want to run the training process must download the required DeepSense 6G Scenario 32 dataset manually and place it in the correct local directory.



\### Scenario 6 - Comparison Experiment



Scenario 6 was used as an additional comparison experiment during the project.

The purpose of Scenario 6 was to compare the main Scenario 32 fusion-based model with another setup, especially for RGB/vision-based analysis.



However, Scenario 6 trained model files were not uploaded to this repository because the model weight files were too large for standard GitHub upload limits.



Therefore:



```text

\- Scenario 6 was used for comparison purposes.

\- Scenario 6 model files are not included in this repository.

\- The main uploaded and maintained experiment is Scenario 32.

```



\## Repository Structure



The repository structure may look like this:



```text

DeepSense6g\_Capstone\_Project/

│

├── dashboard.py

├── best\_multimodal\_beam\_model.pth

├── README.md

├── .gitignore

│

├── scenario32\_training/

│   └── training files

│

└── other project files

```



The exact folder names may change depending on the local project structure.



\## Dataset Requirement



The DeepSense 6G dataset is not included in this repository because of dataset size and usage restrictions.



To run the project, users must manually download the required dataset from the official DeepSense 6G dataset source.



Required dataset:



```text

DeepSense 6G Scenario 32

```



Optional comparison dataset:



```text

DeepSense 6G Scenario 6

```



Recommended local dataset structure:



```text

data/

│

├── scenario32/

│   ├── scenario32.csv

│   ├── unit1/

│   ├── unit2/

│   └── resources/

│

└── scenario6/

&#x20;   ├── scenario6.csv

&#x20;   ├── unit1/

&#x20;   ├── unit2/

&#x20;   └── resources/

```



If the dataset is placed in a different directory, the dataset path inside the Python files must be updated.



\## Environment Setup



It is recommended to create a Python virtual environment before running the project.



\### 1. Create a virtual environment



For Windows:



```bash

python -m venv venv

```



\### 2. Activate the virtual environment



For Windows CMD:



```bash

venv\\Scripts\\activate

```



For Windows PowerShell:



```bash

.\\venv\\Scripts\\Activate.ps1

```



If PowerShell blocks the activation script, run:



```bash

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

```



Then activate the environment again:



```bash

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



If a `requirements.txt` file is available:



```bash

pip install -r requirements.txt

```



If there is no `requirements.txt` file, the main dependencies can be installed manually:



```bash

pip install numpy pandas matplotlib scikit-learn opencv-python torch torchvision streamlit tqdm

```



Additional libraries may be required depending on the training and dashboard scripts.



\## Running the Dashboard



After installing the required libraries and placing the dataset in the correct folder, the dashboard can be run.



If the dashboard is a normal Python script:



```bash

python dashboard.py

```



If the dashboard is built with Streamlit:



```bash

streamlit run dashboard.py

```



\## Running Scenario 32 Training



To run the Scenario 32 training files:



```text

1\. Download the DeepSense 6G Scenario 32 dataset.

2\. Place the dataset under the local data/scenario32/ directory.

3\. Activate the Python virtual environment.

4\. Install the required dependencies.

5\. Update dataset paths in the training script if necessary.

6\. Run the training script.

```



Example:



```bash

python scenario32\_training/train.py

```



The exact command may change depending on the training file name.



\## Model File



The repository includes the trained model file:



```text

best\_multimodal\_beam\_model.pth

```



This file contains the trained model weights for the multimodal beam prediction model. It can be used for testing, evaluation, or dashboard visualization.



\## General Workflow



The general workflow of the project is:



```text

1\. Download the required DeepSense 6G Scenario 32 dataset.

2\. Place the dataset under the data/ directory.

3\. Create and activate a Python virtual environment.

4\. Install the required Python libraries.

5\. Run Scenario 32 training files if training is needed.

6\. Use the trained model for evaluation or visualization.

7\. Run the dashboard to analyze the results.

```



\## Project Goal



The main goal of this capstone project is to analyze beam prediction performance in 6G V2I networks by using deep learning and multimodal sensor data.



Scenario 32 is used as the main multimodal/fusion-based experiment. Scenario 6 was considered as an additional comparison experiment, but it was not uploaded due to large model file sizes.



\## Author



Mustafa Bağlam



\## License



This repository is prepared for academic and educational purposes.



