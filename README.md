# COVID-19 Prediction Model

## Problem Statement
Given Covid-19 data from the European Centre for Disease Prevention and Control for all countires from December 31, 2019 - October 5, 2020 with the following features:
- country_id
- date
- cases
- deaths
- cases_14_100k
- cases_100k

Predict Cvoid-19 deaths in Canada for the following two phases:

**Phase 1**:

October 6, 2020 - October 16, 2020

**Phase 2**:

October 26, 2020 - October 30, 2020

## Machine Learning Model
The Model used for this task consists of a Vector Autoregressive Model with L2 Regularization.

## Data Preprocessing
Wihtin the data, there are spikes of Covid-19 deaths for some days as a result of a delay in reporting. To avoid this spike affecting the model, excess number of Canadian deaths were distributed to previous days.

## Feature Selection
Search and Score algorithm to find optimal combination of features over multiple validation sets.

Countries that the algorithm searched over was restricted to countries with similar levels of economic prosperity and health care infrastructure as Canada.

## Hyperparameter Tuning
Hyperparameters for this model included:
- K: The number of previous days a prediction depends on
- Lambda: L2 Regulatization Hyperparameter
- Days to include in training

Plots were generated for a range of hyperparameters to find combination that yielded lowest error.

## Results
The following errors is the Root Mean Squared Error, showing how close the regression line is to actual points.

**Phase 1**:

17.22049 RMSE

**Phase 2**:

5.47722 RMSE
