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

MODEL_LOCATITON = "raw_data/vg_face_model"

def predict(list_of_arrays_of_images, model_location):
    model = keras.models.load_model(MODEL_LOCATITON)
    list_of_predictions = []
    for image in list_of_arrays_of_images:
        prediction = model.predict(np.expand_dims(tf.image.resize(image, [224, 224]),axis=0)/255.0)
        list_of_predictions.append(prediction[0])
    return list_of_predictions

if __name__ == '__main__':
    image1 = imageio.imread("raw_data/data_3000/train/disgust/17Exp1angry_actor_195.jpg")
    image2 = imageio.imread("raw_data/data_3000/train/disgust/19Exp1angry_actor_198.jpg")
    list_of_arrays_of_images = [image1, image2]

    predictions = predict(list_of_arrays_of_images, MODEL_LOCATITON)
 
    max_location = np.argmax(predictions[0])
    print(f"predictions1 = {CLASSES[max_location]} : {round(predictions[0][max_location]*100)}%")
    print("$$$$$$$$$$")
    max_location = np.argmax(predictions[1])
    print(f"predictions2 = {CLASSES[max_location]} : {round(predictions[1][max_location]*100)}%")
