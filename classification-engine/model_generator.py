from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense, InputLayer
from keras.initializers import RandomNormal
from sklearn.utils import shuffle
from utils import load_image_paths_and_labels, load_images_and_labels

LABELS_FILE_PATH = '../digits/train.txt'
DATASET_PATH = '../digits/train'
SEED = 42

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

images_with_labels = load_image_paths_and_labels(LABELS_FILE_PATH)
images, labels = load_images_and_labels(images_with_labels, DATASET_PATH)
X, y = shuffle(images, labels)
model = build_model(X, y)
model.save('model_v1.h5')
