import streamlit as st
import matplotlib.pyplot as plt
from picturing_emotions.face_detector import DetectFace
import numpy as np
from tensorflow import keras
from tensorflow.image import resize
import cv2
from mtcnn import MTCNN

classes = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]

model = keras.models.load_model("raw_data/vg_face_model")
    


'''

'''
st.markdown('''
# Picturing Emotions
### Please upload a photo to predict the emotions
''')
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose a photo")

detector = MTCNN()

if uploaded_file is not None:
    image = plt.imread(uploaded_file)
    image = cv2.resize(image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    face_locations = detector.detect_faces(image)
    
    for i in range(len(face_locations)):
        x1, y1, width, height =face_locations[i]["box"]
        x2, y2 = x1 + width, y1 + height

        cropped = image[y1:y2, x1:x2]
        pred = model.predict(np.expand_dims(resize((cropped), [224, 224]),axis=0)/255.0)[0]
        top_values = pred.argsort()[-3:]
        prediction1 = classes[top_values[2]]
        pred1       = pred[top_values[2]]
        prediction2 = classes[top_values[1]]
        pred2       = pred[top_values[1]]
        prediction3 = classes[top_values[0]]
        pred3       = pred[top_values[0]]
        

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        cv2.putText(image, f"1: {prediction1}:{round(pred1*100)}%", (x1, y2+30), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image, f"2: {prediction2}:{round(pred2*100)}%", (x1, y2+60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image, f"3: {prediction3}:{round(pred3*100)}%", (x1, y2+90), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    fig, ax = plt.subplots(figsize=(7, 3))
    plt.axis('off') 
    ax.imshow(image)
    st.pyplot(fig) 
    #st.write(pred)
