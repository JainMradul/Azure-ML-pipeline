# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree program.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this project, the UCI Bank Marketing dataset is used to predict whether the bank's clients will open a term deposit with them or not. So, this is a binary classification problem which requires predicting 'yes' or 'no'

For this classification task, broadly two approaches have been used:

> `Scikit-learn` Logistic Regression model is build using Scikit learn library and the hyperparameters are fine tuned using AzureML's Hyperdrive, and this approach is able to achieve an accuracy of 90.85%

> `Azure AutoML` It's a *no code/low code* solution, which used around 26 different ML models to identify the best performing model with an accuracy of almost 91.65%. This approach has a marginal improvement over custom coded logistic regression model.

## Scikit-learn Pipeline
In this approach we have followed below steps,

- [x] Download the dataset in tabular format using TabularDatasetFacory from a specified url
- [x] Pass the data to `clean_code` function for preprocessing and one-hot encoding
- [x] Split the data into `train` and `test` dataset with 80:20 ratio respectively
- [x] Model is trained on the `train` dataset using LogisticRegression estimator which accepts two parameters `C` [regularization strength] and `max_iter` [max itertaion]
- [x] Above hyperparameters are tuned via hyperdrive with `C` in the range of 0 - 0.3 and `max_iter` in the range of 10 - 100

**ParameterSampler**

RandomParameterSampling has been used to give best results within a less amount of time compare to gridsampling which goes through each and every possible combination of the grid and hence consumes a lot of time with almost similar result

**Early Stopping Policy**

BanditPolicy has been used for early stopping to converge quickly when the evaluation metrics is not improving with consecutive runs

## AutoML

Out of 26 models that has been tested by AutoML `Voting Ensemble` turns out to be the best model with an accuracy of 91.65%
The Voting Ensemble model consists of 7 individual models, of which 4 are XGBoost classifiers, and 1 each of a LightGBM, SGD and Logistic Regression
Below config has been used for AutoML,

- [x] experiment_timeout_minutes=30  *save azure resources usage cost*
- [x] task=classification            *since its classification to predict yes/no* 
- [x] primary_metric='accuracy'      *primary evaluation metric to measure model performance*
- [x] n_cross_validations=3          *training on 66% and testing on 33% data to make sure model is not overfitting*

## Pipeline comparison

The AutoML performed better than the SK learn model, even though the accuracy is improved only about 1%. 
Main reason being that AutoML has a variety model at disposal to test out and simply chose the best out of it compare to SK learn approach where we have only tested out the logisticresgression model

Comparing the pipeline, AutoML seems to be low code with best result vs sklearn where more code with comparatively less accurate model

**AutoML run**

![AutoML run](https://github.com/JainMradul/Azure-ML-pipeline/blob/master/automl.PNG)


**Hyperdrive run**

![Hyperdrive run](https://github.com/JainMradul/Azure-ML-pipeline/blob/master/hyperdrive.PNG)

## Future work

- [ ] AutoML (autoguard rails) reveals the class imbalance in data and hence this could be treated using different algo/techniques to improve data quality
- [ ] Depending upon the problem in hand, identify the best evaluation metric that maximize the model performance (Accuracy metric not necessary serves the purpose hence try different metrics as well to evaluate model)
- [ ] AutoML can help you quickly identify the best performing model, hence that model could be coded and used as the base model and further improvement could be done gradually on the top of this model.

## Proof of cluster clean up

![Cluster Clean Up](https://github.com/JainMradul/Azure-ML-pipeline/blob/master/delete.PNG)
