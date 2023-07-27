from tensorflow import keras
import pandas as pd
from keras.utils import to_categorical
from keras.models import load_model


# load csv dataset
def load_dataframe():
  print('Loading data ...')
  dataframe = pd.read_csv(
    DATASET_PATH,
    header=None,
    skiprows=1
  )
  print('Loading data [OK]')

  return dataframe

# split X and y
def split_bases(dataframe):
  print('Split X and y ...')
  X = dataframe.values[:,:-1]

  y = dataframe.values[:, -1]
  y = to_categorical(y)

  print('Split X and y [OK]')
  return X, y

DATASET_PATH = '../digits/val.csv'
MODEL_PATH = './model_v1.h5'

dataframe = load_dataframe()
X, y = split_bases(dataframe)
model = load_model(MODEL_PATH)

result = model.evaluate(X, y)

print(result)
