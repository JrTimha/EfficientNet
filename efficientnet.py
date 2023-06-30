# -*- coding: utf-8 -*-
import keras
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
import os
import glob
from timeit import default_timer as timer


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# In[ ]:
## Import EfficientNet and Choose EfficientNet Model
## loading pretrained conv base model
## define input height and width (B0: 224 | B1: 240 | B2: 260 | B3: 300 | B4: 380 | B5: 456)
width = 299
height = 299
input_shape = (height, width, 3)

# Change Name After .applications.___ for another Pre-Trained Model
conv_base = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Adjust the path:
model_name = "EfficientNetV2"
DATA_BASE_PATH = 'H:\\MachineLearning\\Trainingsdaten'
PATH = f'{DATA_BASE_PATH}/images'
train_dir = PATH+'/train'
valid_dir = PATH+'/test'
test_dir = PATH+'/test'
batch_size = 32

# In[ ]:
# Print dataset informations
dataset_train = os.listdir(train_dir)
print("Classes in this Dataset: ", dataset_train)
print("Number of Classes in this Dataset: ", len(dataset_train))

class_labels = []

for item in dataset_train:
    all_classes = os.listdir(train_dir + '/' + item)

    for room in all_classes:
        class_labels.append((item, str('dataset_train' + '/' + item) + '/' + room))

dataframe = pd.DataFrame(data=class_labels, columns=['Labels', 'Image'])
print(dataframe.head())
print(dataframe.tail())
print("Total Number of Images in this Dataset: ", len(dataframe))
# In[ ]:
# Section 1: Preprocessing
# DO NOT rescale EfficientNet!
# Preprocessing function with Data Augmentation:
train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rotation_range=20,
                                   zoom_range=0.15,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.15,
                                   horizontal_flip=True,
                                   # vertical_flip=True,
                                   fill_mode="nearest",
                                   preprocessing_function=tf.keras.applications.xception.preprocess_input
                                   )  # 80% for Training and 20% for Validation

# Images used for training the model
train_generator = train_datagen.flow_from_directory(
    train_dir,                                      # This is the target directory
    target_size=(width, height),                    # All images will be resized to the needed target height and width.
    batch_size=batch_size,                          # Standard: 32
    class_mode='categorical')                       # We use categorical_crossentropy loss and need categorical labels

# Images used for the validation accuracy
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='categorical')

# In[ ]:
# Section 2: CNN-Modelling

epochs = 10  # Change this for longer training
NUM_TRAIN = sum([len(files) for r, d, files in os.walk(train_dir)])
NUM_TEST = sum([len(files) for r, d, files in os.walk(valid_dir)])
dropout_rate = 0.0

num_classes = len(os.listdir(train_dir))
print('building network for ' + str(num_classes) + ' classes')

# _______________________________________________________________________________________
"""---------Building the Model---------"""
for layer in conv_base.layers:
    layer.trainable = False
conv_base.summary()
model = models.Sequential()                                                 # Building CNN with Sequential-Class
model.add(conv_base)                                                        # Adding the Pre-Trained Model
model.add(layers.GlobalAveragePooling2D(name="gap"))
# Extract the richest feature
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))             # Dropout, avoid overfitting
model.add(layers.Dense(num_classes, activation='softmax', name="fc_out"))   # Softmax-Classification for 21 classes
model.summary()                                                             # Printing all layers and metrics
"""-----------End Building-----------"""

print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))

print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))

# In[ ]:
# Section 3: Training

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['acc'])

history = model.fit_generator(                      # Starting the training algorithm
    train_generator,                                # Streaming the trainingset in batches
    steps_per_epoch=NUM_TRAIN // batch_size,
    epochs=epochs,
    validation_data=validation_generator,           # Streaming the validationset in batches
    validation_steps=NUM_TEST // batch_size,
    verbose=1,
    use_multiprocessing=False)

# In[ ]:
# Fine Tuning EfficientNet
save_path = f'./models/{model_name}10.h5'
os.makedirs("./models", exist_ok=True)
model.save(save_path)

for layer in conv_base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

#for layer in range(-26, 0, 1):
    #if not isinstance(conv_base.layers[layer], layers.BatchNormalization):
       # conv_base.layers[layer].trainable = True
       # print(conv_base.layers[layer].name)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=NUM_TRAIN // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=NUM_TEST // batch_size,
    verbose=1,
    use_multiprocessing=False)

# In[ ]:
# Section 4: Saving model and metrics

# Get staticstics about training results:
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_x, loss, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

"""# Save Tensorflow Model"""
f'{DATA_BASE_PATH}/images'
save_path = f'./models/{model_name}.h5'
os.makedirs("./models", exist_ok=True)
model.save(save_path)

# In[ ]:
# Section 5: Testing

# Stream image batches of the testset:
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(width, height),
    batch_size=1,
    class_mode='categorical')
new_model = tf.keras.models.load_model("./models/EfficientNetV2.h5")
# Start evaluation to test acc in practical use:
metrics = new_model.evaluate(test_generator)
print("%s: %.2f%%" % (model.metrics_names[1], metrics[1]*100))

test_path = PATH+'/Benchmark'
image_list = []
# transform the test images to tensors:
for filename in glob.glob(test_path+'/*.jpg'):
    img = load_img(filename, target_size=(width, height))
    img = img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    image_list.append(img)

i = 0
time_calc = 0
# test time per inference:
while i < 5:
    start = timer()
    for device in image_list:
        pred = new_model.predict(device)
    end = timer()
    time_calc = time_calc + (((end - start) / 100) * 1000)
    i += 1
print("Zeit pro Bild: " + str(time_calc / 5) + "ms")
