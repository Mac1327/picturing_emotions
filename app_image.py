import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.image import resize
import cv2
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)



#settings for webrtc to stop audio
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
#classes for predcitons
classes = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]

#load the trained emtiondetector model
model = keras.models.load_model("raw_data/vg_face_model")
#load the face detector xml
haar_cascade = cv2.CascadeClassifier("haar_face.xml")



st.markdown('''
# Picturing Emotions
### Press START to capture your emotions.
''')

class VideoTransformer(VideoTransformerBase):
    #inherit from streamlit_webrtc transformer
    def transform(self, frame):
        #convert mesh to array using bgr as this works best with cv2
        frame = frame.to_ndarray(format="bgr24")
        
        #locate all faces in the array image 
        faces_box = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=11)

        for (x1,y1,w,h) in faces_box:
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), thickness=2)      
            
            #select only the face from the image and predict emotions with the imported model
            cropped = cv2.resize(frame[y1:y2, x1:x2], dsize=(224, 224), interpolation=cv2.INTER_CUBIC) /255.0
            
            #make prediciton with model
            pred = model.predict(np.expand_dims(cropped,axis=0))[0]

            #get the top 3 vlause and their class name
            top_values = pred.argsort()[-3:]
            prediction1 = classes[top_values[2]]
            pred1       = pred[top_values[2]]
            prediction2 = classes[top_values[1]]
            pred2       = pred[top_values[1]]
            prediction3 = classes[top_values[0]]
            pred3       = pred[top_values[0]]

            #wirte the the top 3 predictions above each box as precitons
            # COLOURS!!! All colours are BRG not RGB   
            cv2.putText(frame, f"1: {prediction1}:    {round(pred1*100)}%", (x1, y1-40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
            cv2.putText(frame, f"2: {prediction2}:    {round(pred2*100)}%", (x1, y1-25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
            cv2.putText(frame, f"3: {prediction3}:    {round(pred3*100)}%", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        #retunr the image with all boxes and predictions on top 
        return frame


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS)