# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
# import the models for further classification experiments
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3
)

# =============================================================================
# Initialize the models
# =============================================================================
vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

models = [vgg_model, inception_model, resnet_model, mobilenet_model]
models_names = ['vgg_model', 'inception_model', 'resnet_model', 'mobilenet_model']

# =============================================================================
# Predict Method
# =============================================================================


def predict_bread(filenames):
    processed_images = []
    for filename in filenames:
        original = load_img(filename, target_size=(224, 224))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        processed_image = vgg16.preprocess_input(image_batch.copy())
        processed_images.append(processed_image)

    for model in models:
        print("using model: ", model.name)
        predictions = model.predict(processed_images)
        label_vgg = decode_predictions(predictions)

        for prediction_id in range(len(label_vgg[0])):
            print(label_vgg[0][prediction_id])


# =============================================================================
# Specify .jpg files
# =============================================================================
jake = ['Jake/jake1.jpg', 'Jake/jake2.jpg', 'Jake/jake3.jpg', 'Jake/jake4.jpg']
ermis = ['Ermis/Ermis1.jpg', 'Ermis/Ermis2.jpg', 'Ermis/Ermis3.jpg']
fred = ['Fred/Fred1.jpg', 'Fred/Fred2.jpg', 'Fred/Fred3.jpg']

# =============================================================================
# Calling Method
# =============================================================================
print("\nJake predictions: ")
predict_bread(jake)
print("\nErmis predictions: ")
predict_bread(ermis)
print("\nFred predictions: ")
predict_bread(fred)
