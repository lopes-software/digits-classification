from keras.utils import to_categorical
from keras.models import load_model
from utils import load_image_paths_and_labels, load_images_and_labels

LABELS_FILE_LPATH = '../digits/test.txt'
DATASET_PATH = '../digits/test'
MODEL_PATH = './model_v1.h5'

images_with_labels = load_image_paths_and_labels(LABELS_FILE_LPATH)
X, y = load_images_and_labels(images_with_labels, DATASET_PATH)
model = load_model(MODEL_PATH)

result = model.evaluate(X, y)

print(result)
