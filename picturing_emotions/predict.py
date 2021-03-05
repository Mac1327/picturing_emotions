import tensorflow as tf 
from tensorflow import keras
import numpy as np
import imageio

CLASSES = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]


def predict(list_of_arrays_of_images, model_location):
    model = keras.models.load_model(model_location)
    list_of_predictions = []
    for image in list_of_arrays_of_images:
        prediction = model.predict(np.expand_dims(tf.image.resize(image, [224, 224]),axis=0)/255.0)
        list_of_predictions.append(prediction[0])
    return list_of_predictions

if __name__ == '__main__':
    model_location = "raw_data/vg_face_model"
    image1 = imageio.imread("raw_data/data_3000/train/anger/AF01ANS.png")
    image2 = imageio.imread("raw_data/data_3000/train/fear/AF02AFS.png")
    list_of_arrays_of_images = [image1, image2]
    predictions = predict(list_of_arrays_of_images, model_location)
    print(f"predictions1 ={CLASSES[np.argmax(predictions[0])]}, predictions2 ={CLASSES[np.argmax(predictions[2])]}")