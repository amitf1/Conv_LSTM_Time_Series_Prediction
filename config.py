import logging
from datetime import datetime
import sys

APP_PATH = "https://fake-app.herokuapp.com"
BATCH_SIZE = 64
BUFFER_SIZE = 10000
HIST_SIZE = 96*2
TARGET_SIZE = 96
SUBSEQ_N = 4
MEAN_STD_PATH = 'train_mean_std.csv'
KEY = 'key.json'

logger = logging.getLogger("CNN_LSTM_MODEL_LOG")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f'CNN_LSTM_MODEL_{datetime.now()}.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler(sys.stdout))
