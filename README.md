# Code Walkthrough
The code contains the following components:
1. data - This directory contains the original dataset, as well as any of the additional datasets, and models created
as part of the training process.
2. experiments - This directory contains a jupyter notebook that was used for exploratory data analysis.
3. src - This package contains all the source code.
4. Pipfile - This is the file with all dependencies.
5. Pipfile.lock - This is the lock file created using Pipfile.
6. README.md - This is the instruction manual.

# Steps to run the code
1. Firstly, create a Pipenv environment, and install dependencies mentioned in Pipfile / Pipfile.lock. 
2. Next, activate the environment, and run `run_train.py --xgb --test` to train and run the best model, the XGBoost model. Other 
models can be run by changing the `--xgb` flag to other flags mentioned in the same file.