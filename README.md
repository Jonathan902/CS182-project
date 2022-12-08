# Li-ion Battery Estimation

## Data Acquisition
To obtain the data that used to train and test the models, run [this simulation notebook](simulation.ipynb). <br /> 
This notebook runs the simulation to generate data, preprocesses the data into time series, and split them into training and testing set. <br /> 
The csv files can be accessed here: [training set](train.csv) and [testing set](test.csv) <br /> 

### Dataset overview:
Input: ${V}$, ${I}$, ${T_s}$, ${prev\_SoC}$, ${prev\_Tc}$<br /> 
Primary target: ${SoC}$ <br /> 
Auxiliary target: ${T_c}$ (see model 5 - 8 below)<br /> 
Data are preprocessed to different timesteps through [seq_data](seq_data.py)<br /> 

## Experiment
To reproduce the experiment done in this project, run [the Experiment notebook](Experiments.ipynb) <br /> 
In this notebook, we train: <br /> 
### Baseline Models 
1. Vanilla RNN  (single layer) <br /> 
2. Vanilla LSTM  (single layer) <br /> 

### Experiment Models

3. Vertically-Stacked RNN <br /> 
4. Vertically-Stacked LSTM <br /> 
5. Vanilla RNN with Tc as auxiliary target (single layer) <br /> 
6. Vanilla LSTM with Tc as auxiliary target (single layer) <br /> 
7. Vertically-Stacked RNN with Tc as auxiliary target <br /> 
8. Vertically-Stacked LSTM with Tc as auxiliary target <br /> 

*The implementation of training and testing loops are in [utils.py](utils.py)*

### Performance Evaluation
At the end of the notebook, model performance is evaluated using the mse loss function <br /> 
Visualization of loss over epoch for different models are also presented in the experiment notebook <br /> 

#### Reference
https://www.crosstab.io/articles/time-series-pytorch-lstm/