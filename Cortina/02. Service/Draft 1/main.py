import io
import logging
import os

import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from keras.preprocessing.image import img_to_array
from model.load import init_model

# Load app
app = Flask(__name__)

# Load modell
global model, graph


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.true_divide(image, 255)
    image = np.expand_dims(image, axis=0)

    # return the processed image
    return image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("file"):
            # read the image in PIL format
            file = request.files["file"]
            image = file.read()
            image = Image.open(io.BytesIO(image))

            # save the image to the static folder
            folder = os.path.basename('static')
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                y_prob = model.predict(image)

                classes = ['Ballerinas',
                           'Booties',
                           'Boots',
                           'Houseshoes',
                           'Low shoes',
                           'Outdoor',
                           'Pumps',
                           'Sandals and Mules',
                           'Sneakers',
                           'Sport']

                y_classes = list(zip(classes, y_prob[0]))
                sorted_by_second = sorted(y_classes, key=lambda tup: tup[1], reverse=True)

                sorted_by_second_plus10 = [x for x in sorted_by_second if x[1] >= 0.1]

                probstring = ""
                for s in sorted_by_second_plus10:
                    probstring += s[0] + " (" + str(s[1]*100)[:4]  +"%)"
                    probstring += "  -  "

                probstring = probstring.strip("  -  ")


        return probstring


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


def setup_app(app):
    global model, graph
    model, graph = init_model()


setup_app(app)

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.

    app.run(host='127.0.0.1', port=5000, debug=True)
# [END app]
