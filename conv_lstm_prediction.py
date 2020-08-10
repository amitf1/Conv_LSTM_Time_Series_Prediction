import json
import os
import config as CNFG
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
from kando import kando_client
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPooling1D, Dense, LSTM, TimeDistributed, Conv1D, Flatten


class ConvLSTM:
    def __init__(self, sensor, trained_model_path=None):
        """
        :param sensor: name of sensor to predict e.g 'EC', 'PH'
        :param trained_model_path: load a pre trained model to use as starting point for training or
         to use for prediction

        """
        try:
            with open(CNFG.KEY) as f:
                secret = json.load(f)
        except FileNotFoundError:
            CNFG.logger.error(f"Please attach json with key & secret, no file named key.json found")

        self._client = kando_client.client(CNFG.APP_PATH, secret['key'], secret['secret'])
        CNFG.logger.info("API Connected")
        self.sensor = sensor.upper()

        if trained_model_path:
            self._model = load_model(trained_model_path)
        else:
            self._model = self._build_model()
        self.train_mean = None
        self.train_std = None

    def _get_data(self, point_id, start, end):
        """
        Get data from the API
        :param point_id: id of the sensor's location point
        :param start: start time of the wanted data
        :param end: end time of the wanted data
        :return: dictionary with point data and it's samplings
        """
        return self._client.get_all(point_id, '', start, end)

    def _preprocess(self, point_id, start, end):
        """
        Process, standardize and fill missing values in the data
        :param point_id: id of the sensor's location point
        :param start: start time of the wanted data
        :param end: end time of the wanted data
        :return: process series and target series to use for labels as well as train std for MAE calculation
        """
        data = self._get_data(point_id, start, end)
        df = pd.DataFrame(data['samplings']).T
        df.index = df.DateTime.astype('int').astype("datetime64[s]")
        series = df[self.sensor].astype('float64')
        series = series.resample('15T').pad()
        CNFG.logger.info(f"Training first sample {df.index[0]}, Training last sample {df.index[-1]}")
        series[series == 0] = np.nan
        series.interpolate(limit_direction='both', inplace=True)
        train_mean = series.mean()
        train_std = series.std()
        self.train_mean = train_mean
        self.train_std = train_std
        pd.DataFrame({"mean": {0: train_mean}, "std": {0: train_std}}).to_csv(
            f'{self.sensor}_{point_id}_{CNFG.MEAN_STD_PATH}')
        CNFG.logger.info(f"Before normalization - Min value {series.min()}, Max value {series.max()}")
        series = (series - train_mean) / train_std
        CNFG.logger.info(f"After normalization - Min value {series.min()}, Max value {series.max()}")
        series = series.clip(-3, 3)
        assert series.isna().sum() == 0
        target = series.copy()
        series = series.values.reshape(-1, 1)
        return series, target, train_std

    @staticmethod
    def _create_windowed_dataset(dataset, target, start_index, end_index, history_size,
                                 target_size, step, single_step=False):
        """
        Creates windows from the given dataset, training windows with history size and the next window to predict
        with target size
        :param dataset: data to be transformed into windows
        :param target: labels for the dataset - the next window to predict
        :param start_index: first index to include in the windows
        :param end_index: last index to include in the windows, if None, then the end of the dataset is the last index
        :param history_size: the size of the past window of information
        :param target_size: how far in the future does the model need
         to learn to predict, the size of the labels
        :param step: how many data samples between two used samples
        :param single_step: whether to predict the whole window with target size - False, or just the end value - True
        :return: dataset with windows for features and labels as numpy arrays
        """
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = np.arange(i - history_size, i, step)
            data.append(dataset[indices])

            if single_step:
                labels.append(target[i + target_size])
            else:
                labels.append(target[i:i + target_size])

        return np.array(data), np.array(labels)

    @staticmethod
    def _build_model():
        """
        building a Keras model consists of CNN and LSTM layers
        :return: compiled Keras model
        """
        time_steps_per_seq = int(CNFG.HIST_SIZE // CNFG.SUBSEQ_N)
        multi_step_model = Sequential(
            [
                TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                                input_shape=(CNFG.SUBSEQ_N, time_steps_per_seq, 1)),
                TimeDistributed(MaxPooling1D(pool_size=2)),
                TimeDistributed(Flatten()),
                LSTM(32, return_sequences=True),
                LSTM(16, activation='relu'),
                Dense(CNFG.TARGET_SIZE)
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        loss = tf.keras.losses.Huber()
        multi_step_model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
        CNFG.logger.info(f"Model was built and compiled")
        return multi_step_model

    def fit(self, point_id, start, end):
        """
        train the model on the data according to given point and time frame
        :param point_id: id of the sensor's location point
        :param start: start time of the wanted data
        :param end: end time of the wanted data
        """
        series, target, std = self._preprocess(point_id, start, end)
        time_steps_per_seq = int(CNFG.HIST_SIZE // CNFG.SUBSEQ_N)
        X_train, y_train = self._create_windowed_dataset(series, target, 0, None, CNFG.HIST_SIZE, CNFG.TARGET_SIZE, 1)
        # split each window into subsequences to be used by a convolution layer
        X_train = X_train.reshape((X_train.shape[0], CNFG.SUBSEQ_N, time_steps_per_seq, X_train.shape[-1]))
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data = train_data.cache().shuffle(
            CNFG.BUFFER_SIZE).batch(CNFG.BATCH_SIZE).prefetch(1)
        multi_step_history = self._model.fit(train_data, epochs=15)
        CNFG.logger.info(f"Train MAE: {np.round(std*multi_step_history.history['mae'][-1], 2)}")
        model_file_name = f'{self.sensor}_{point_id}_CNN_LSTM.h5'
        self._model.save(model_file_name)
        CNFG.logger.info(f"Model Saved as {model_file_name}")

    def predict(self, point_id, start):
        """
        Use the fitted model to predict values for given point from given start time
        :param point_id: id of the sensor's location point
        :param start: start time of the wanted prediction

        :return: prediction values for the given point starting from the given starting time
        """
        mean_std_file = f'{self.sensor}_{point_id}_{CNFG.MEAN_STD_PATH}'
        if self.train_mean is None or self.train_std is None:
            if os.path.exists(mean_std_file):
                std_mean = pd.read_csv(mean_std_file)
                train_mean = std_mean.loc[0, 'mean']
                train_std = std_mean.loc[0, 'std']

            else:
                CNFG.logger.error(f"Model instance should contain train_mean and train_std")
                raise ValueError("Model instance should contain train_mean and train_std but at least one of them "
                                 f"is None. No file with name {mean_std_file} containing "
                                 "previous values exists or used."
                                 "\nEither train the model by calling fit, or use previous training values file")
        else:
            train_mean, train_std = self.train_mean, self.train_std

        # start and end time for the 2 days of samples prior to prediction day
        history_end = start - 15
        history_start = history_end - datetime.timedelta(days=2).total_seconds()
        X = self._get_data(point_id, history_start, history_end)
        df = pd.DataFrame(X['samplings']).T
        df.index = df.DateTime.astype('int').astype("datetime64[s]")
        series = df[self.sensor].astype('float64')
        series = series.resample('15T').pad()
        series[series == 0] = np.nan
        series.interpolate(limit_direction='both', inplace=True)
        series = (series - train_mean)/train_std
        X = series.values
        time_steps_per_seq = int(CNFG.HIST_SIZE // CNFG.SUBSEQ_N)
        X = X.reshape(1, CNFG.SUBSEQ_N, time_steps_per_seq, 1)
        return self._model.predict(X)*train_std + train_mean
