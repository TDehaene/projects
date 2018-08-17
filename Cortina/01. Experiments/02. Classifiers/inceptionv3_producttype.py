import pickle
import time

from keras import applications
from keras import backend as K
from keras import optimizers
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint, History, TensorBoard, EarlyStopping
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224
train_data_dir = r"C:\Users\peter\Documents\GitHub\projects\Cortina\00. Data\ProductType/train"
validation_data_dir = r"C:\Users\peter\Documents\GitHub\projects\Cortina\00. Data\ProductType/test"
TARGET_SIZE = (img_width, img_height)
BATCH_SIZE = 8
epochs = 2

clear_session()


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    base_model = applications.ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(100, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    print(model.summary())

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
                    metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    vertical_flip=False,
    zoom_range=0
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

history = History()

earlystopping = EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=1)
filepath = "weights-pretrainedvgg16-minus6-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

model.fit_generator(
    train_datagen.flow_from_directory(train_data_dir, target_size=TARGET_SIZE),
    epochs=epochs,
    steps_per_epoch=6731 / BATCH_SIZE,
    validation_data=val_datagen.flow_from_directory(validation_data_dir,
                                                    target_size=TARGET_SIZE),
    validation_steps=1684 / BATCH_SIZE,
    callbacks=[history, checkpoint, earlystopping]
)

# Save history

time.sleep(10)

print(history.history)

with open("history_history.pickle", "wb") as file:
    pickle.dump(history.history, file)

with open("history_model.pickle", "wb") as file:
    pickle.dump(history.model, file)
