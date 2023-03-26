# CI/CD Pipeline with DVC and CML

This repository contains a sample project using [CML](https://github.com/iterative/cml) with [DVC](https://github.com/iterative/dvc) to push/pull data from cloud storage and track model metrics. When a pull request is made in this repository, the following will occur:
- GitHub will deploy a runner machine and install all requirements
- DVC will pull data from cloud storage
- The runner will execute a workflow to train a ML model (`python train.py`)
- A visual CML report about the model performance with DVC metrics will be returned as a comment in the pull request

The key file enabling these actions is `.github/workflows/cml.yaml`.

## Secrets and environmental variables
In this example, `.github/workflows/cml.yaml` contains two environmental variables that are stored as repository secrets.

| Secret  | Description  | 
|---|---|
|  GITHUB_TOKEN | This is set by default in every GitHub repository. It does not need to be manually added.  |
|  $GOOGLE_API_PW  | Passphrase to decode the json containing google's application credentials.  | 

We used DVC with Google Drive which is why we need Google credentials.

## Project content
This project is related to image classification on MNIST dataset. 
We pull the dataset from DVC and train a simple classifier on it. 

Our training script returns a [metrics file](metrics.json) along with a [csv tracking the loss function](loss.csv). The reference for both of these files is present on the main branch. 

For each PR, these files are regenerated with new values and we use `dvc metrics diff` along with `dvc plots diff` to compare the new values to the references from the main branch.
