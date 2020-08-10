# CNN LSTM Time Series Prediction and Anomaly Detection
## By Amit Feldman

Predicting Iot Sensors' values, using CNN LSTM Time Series model.
The model was developed to find patterns and predict future values for sensors that are measuring chemical values of water, such as EC, PH etc.
Each one of the predictions is made for one day - 96  samples(every 15 minutes), and values far from the prediction by a certain threshold are considered anomalies.

The model can handle a gap from the training phase (it doesn't have to be trained until the point before the predictions like other time series models).
The Neural Network consists of CNN and LSTM layers, while the data is split into windows of 2 days as X and one day as label y.

EC_CNN_LSTM_5m_2020_Full_Day-BS64_Refaiim.ipynb shows the development process.
Example Using CNN_LSTM Model.ipynb shows how to use the module, and anlysis of the results.
conv_lstm_prediction.py is the code for the module.

This project for now is only designed for using an internal API which is is private, an option to use a json file with other data will be up later on.
