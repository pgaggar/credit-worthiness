# Code Walkthrough
The code contains the following components:
1. data - This directory contains the original dataset, as well as any of the additional datasets, and models created
as part of the training process. It contains `images` subdirectory, which contains the relevant plots for each model.
2. experiments - This directory contains a jupyter notebook that was used for exploratory data analysis.
3. src - This package contains all the source code. It contains the trainer package, which contains all ML trainers.
The `__init__` file contains the analysis and experiments carried out with each model. It also contains `utils` package which 
contains `load_and_process.py` to load and preprocess the data. It also contains the `non_ml` package, which contains the non-ml classifier.
4. Pipfile - This is the file with all dependencies.
5. Pipfile.lock - This is the lock file created using Pipfile.
6. README.md - This is the instruction manual.

# Steps to run the code
1. Firstly, create a Pipenv environment, and install dependencies mentioned in Pipfile / Pipfile.lock / requirements.txt. 
2. Next, activate the environment, and run `run_train.py --xgb --test` to train and run the best model, the XGBoost model. Other 
models can be run by changing the `--xgb` flag to other flags mentioned in the same file.
