import cv2
import numpy as np

# Loads image labels file and returns an Map with its related values.
#
# @param labels_file_path [String]: path to labels file.
#
# @returns: an Map with image file name as key and its class as value.
def load_image_paths_and_labels(labels_file_path):
  print('Load data...')
  images_with_labels = {}

  with open(labels_file_path) as labels_file:
    for line in labels_file:
      splitted_values = line.split(' ')
      image_name = splitted_values[0]
      label = splitted_values[1].strip()
      images_with_labels[image_name] = label

  print('Load data [OK]')
  return images_with_labels


# Loads image dataset and returns an array with normalized images and an second array with images
#   respective labels.
#
# @param images_with_labels [Map]: map where keys are image file name and values are image label.
# @param images_base_path [String]: path to directory with images.
#
# @returns: an numpy array with normalized images and an second array with respective labels.
def load_images_and_labels(images_with_labels, images_base_path):
  labels = []
  images = []

  for key in images_with_labels.keys():
    image_path = f'{images_base_path}/{key}'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # load image on grayscale
    normalized_image = image / 255.0                     # normalize values
    images.append(normalized_image)
    labels.append(images_with_labels[key])
    print('.', end='')

  print()
  numpy_images = np.array(images, np.uint8)
  numpy_images = numpy_images.reshape(-1, 64 * 64)

  return numpy_images, labels
