import streamlit as st
from picturing_emotions.face_detector import DetectFace
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
from mtcnn import MTCNN

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)


classes = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]

model = keras.models.load_model("raw_data/vg_face_model")
    
detector = MTCNN()

class VideoTransformer(VideoTransformerBase):
    
    def transform(self, frame):
        #imgin = frame.to_ndarray(format="bgr24")
        #img = cv2.cvtColor(imgin, cv2.COLOR_BGR2RGB)


        
        face_locations = detector.detect_faces(frame)
        x1, y1, width, height =face_locations[0]["box"]
        x2, y2 = x1 + width, y1 + height

        cropped = frame[y1:y2, x1:x2]
        pred = model.predict(np.expand_dims(resize(cropped, [224, 224]),axis=0)/255.0)[0]


        top_values = pred.argsort()[-3:]
        prediction1 = classes[top_values[2]]
        pred1       = pred[top_values[2]]
        prediction2 = classes[top_values[1]]
        pred2       = pred[top_values[1]]
        prediction3 = classes[top_values[0]]
        pred3       = pred[top_values[0]]

                
        cv2.putText(frame, f"1: {prediction1}:    {round(pred1*100)}%", (x1, y1-40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, f"2: {prediction2}:    {round(pred2*100)}%", (x1, y1-25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        cv2.putText(frame, f"3: {prediction3}:    {round(pred3*100)}%", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS)