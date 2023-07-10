import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import keras_cv
import numpy as np
from keras_cv import bounding_box
from keras_cv import visualization
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

pretrained_model = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc", bounding_box_format="xywh"
)

image_paths = [
    "https://imgv3.fotor.com/images/side/generate-cat-dog-and-birds-with-fotor-random-animal-generator.jpg",
]

images = []
for path in image_paths:
    filepath = tf.keras.utils.get_file(origin=path)
    image = keras.preprocessing.image.load_img(filepath, target_size=(640, 640))
    image = keras.preprocessing.image.img_to_array(image)
    images.append(image)

images = np.array(images)

# visualization.plot_image_gallery(
#     images,
#     value_range=(0, 255),
#     rows=1,
#     cols=1,
#     scale=5,
# )

inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)

image_batch = inference_resizing([image])

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]

class_mapping = dict(zip(range(len(class_ids)), class_ids))

y_pred = pretrained_model.predict(image_batch)
# y_pred is a bounding box Tensor:
# {"classes": ..., boxes": ...}
visualization.plot_bounding_box_gallery(
    image_batch,
    value_range=(0, 255),
    rows=1,
    cols=1,
    y_pred=y_pred,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_mapping,
)
