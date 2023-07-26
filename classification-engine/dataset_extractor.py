import cv2
import pandas as pd

def load_images_and_labels():
  print('Load data...')
  images_with_labels = {}

  with open(LABELS_FILE_PATH) as labels_file:
    for line in labels_file:
      splitted_values = line.split(' ')
      image_name = splitted_values[0]
      label = splitted_values[1].strip()
      images_with_labels[image_name] = label

  print('Load data [OK]')
  return images_with_labels

def build_dataset(images_with_labels):
  print('Build dataset')
  dataset = []

  for key in images_with_labels.keys():
    image_path = f'../digits/train/{key}'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # load image on grayscale
    normalized_image = image.flatten() / 255.0           # flatten and normalize values
    features = normalized_image.tolist()                 # convert into a list
    label = images_with_labels[key]
    features.append(label)                               # append label
    dataset.append(features)
    print('.', end='')

  print()
  print('Build dataset [OK]')
  return dataset


# NOTE: base file name
LABELS_FILE_PATH = '../digits/train.txt'
OUTPUT_FILE_PATH = '../digits/train.csv'


images_with_labels = load_images_and_labels()
dataset = build_dataset(images_with_labels)
print('Saving CSV file ...')
dataframe = pd.DataFrame(data=dataset)
dataframe.to_csv(f'{OUTPUT_FILE_PATH}')
print('Saving CSV file [OK]')
