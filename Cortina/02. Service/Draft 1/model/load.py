import os

import tensorflow as tf
from keras.models import model_from_json


def init_model():
    # load model structure from json
    json_file = open(os.path.abspath('model/model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # load weights into new model
    loaded_model.load_weights(os.path.abspath('model/model_weights.h5'))

    graph = tf.get_default_graph()

    return loaded_model, graph
