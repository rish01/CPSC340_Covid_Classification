# CPSC340 Final Assignment Part 2 - COVID-19 Lung X-Ray Classification

The purpose of this part of the assignment is to classify whether a given X-ray of a lung is from a patient with COVID-19 or not. The final report of this task is present [here](https://github.com/rish01/CPSC340_Covid19_Lung_Classification/blob/master/report/q2_covAId.pdf) and it contains description of given problem, machine learning models developed by me and my team, and results.

## Installation
1. Clone the repo on your system. 
2. Install Anaconda from [here](https://www.anaconda.com/products/individual) which will be used to setup virtual environment for the project.
3. Ensure conda is installed and in the PATH in your system's environment variables. 
4. Create a virtual environment using conda by typing the entering this command in your terminal/command prompt: <br />
```conda create -n covid19_classification_venv python=3.6 anaconda```
5. Activate the newly created virtual environment using this command:<br />
```source activate covid19_classification_venv```
6. Navigate to the folder containing the repo (if not already there).
7. Install the required Python packages to the virtual environment using this command:<br />
```conda install -n covid19_classification_venv -r requirements.txt```
8. Link the python.exe file present in the conda virtual environment folder to the interpreter of your IDE. 

<br />

## Testing a Model
Before running main.py file, please change the value of *model_to_run* variable present on line 32 to one of *KERAS_CNN, TRANSFER_LEARNING or FULLY_CONNECTED_LINEAR*.
