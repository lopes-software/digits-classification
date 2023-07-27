import pandas as pd
from keras.utils import to_categorical
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, InputLayer
from keras.initializers import RandomNormal
from sklearn.utils import shuffle

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

  X, y = shuffle(X, y) # without shuffle data accuracy on test was 0.7 and after shuffle it grows to 0.97

  print('Split X and y [OK]')
  return X, y

def build_model(X, y):
  print('Training model ...')
  model = Sequential([
    InputLayer(input_shape=(4096,)),
    Dense(256, activation='relu', kernel_initializer=RandomNormal()),
    Dense(128, activation="sigmoid", kernel_initializer=RandomNormal()),
    Dense(10, activation='softmax')
  ])

  model.compile(
    loss="categorical_crossentropy",
    optimizer="rmsprop",
    metrics=["categorical_accuracy"]
  )

  model.fit(
    X,
    y,
    epochs=50,
    validation_split=0.2
  )
  print('Training model [OK]')

  return model


DATASET_PATH = '../digits/train.csv'
SEED = 42


dataframe = load_dataframe()
X, y = split_bases(dataframe)
model = build_model(X, y)
model.save('model_v1.h5')
