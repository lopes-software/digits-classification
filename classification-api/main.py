from flask import Flask, jsonify, request
import numpy as np
from tensorflow import keras
from keras.models import load_model
# NOTE: keras try to use all memory aviable, and it results on chash when there is other processes
#   using GPU at the same time.
import tensorflow as tf

config = tf.compat.v1.ConfigProto(
  gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

model = load_model('./model_v1.h5')

app = Flask(__name__)

@app.route('/classify_digit', methods=['POST'])
def classify_digit():
  payload = request.get_json()

  image = np.array([payload['image']]) # np.array must receive an array of arrays
  image = image / 255.

  digits_probabilities = model.predict(image)[0].tolist()
  digit = digits_probabilities.index(max(digits_probabilities))

  return jsonify(digit=digit, digits_probabilities=digits_probabilities)

app.run(debug=True)
