import streamlit as st
import matplotlib.pyplot as plt
from picturing_emotions.face_detector import DetectFace
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.image import resize
import cv2
from mtcnn import MTCNN
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

classes = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]

model = keras.models.load_model("raw_data/vg_face_model")

st.sidebar.markdown(f"""
    # Picturing Emotions
    """)



image = Image.open('Logo.png')
st.image(image, caption=' ', use_column_width=False)

# st.markdown("""
# <style>
# body {
#     color: #000000;
#     background-color: #f4d160;
# }
# </style>
#     """, unsafe_allow_html=True)



st.markdown(
    f"""
    <style>
    .reportview-container {{
         background-color: #f4d160;
    }}
   .sidebar .sidebar-content {{
        background-color: red;
    }}
    </style>
    """,
    unsafe_allow_html=True
 )



### Team profile
#st.title('Meet our team')
greeting = f"""
<hr>
<h1><strong><span style="color: rgb(239, 239, 239);">Meet our team</span></strong></h1>
<hr>
<p><br></p>
"""
st.write(greeting, unsafe_allow_html=True)

team = Image.open('Team.png')
st.image(team, use_column_width=False)



HTML1 = f"""
<hr>
<h4 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">Add text later.</span></h2>
<hr>
<p><br></p>
"""
st.write(HTML1, unsafe_allow_html=True)

'''



'''

HTML2 = f"""
<h2 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">Please upload a photo to predict the emotions.</span></h2>
"""
st.write(HTML2, unsafe_allow_html=True)



uploaded_file = st.file_uploader("Choose a photo")

detector = MTCNN()

if uploaded_file is not None:
    image = plt.imread(uploaded_file)
    image = cv2.resize(image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
    image_origin = image.copy()

    face_locations = detector.detect_faces(image)

    pred_values = {}
    X1, X2, Y1, Y2 = [],[],[],[]

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
        

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv2.putText(image, f"1: {prediction1}:{round(pred1*100)}%", (x1, y2+30), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image, f"2: {prediction2}:{round(pred2*100)}%", (x1, y2+60), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(image, f"3: {prediction3}:{round(pred3*100)}%", (x1, y2+90), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        pred_values[i] =  [[prediction1, round(pred1*100)],
                           [prediction2, round(pred2*100)],
                           [prediction3, round(pred3*100)]]

        X1.append(x1)
        X2.append(x2)
        Y1.append(y1)
        Y2.append(y2)

    fig, ax = plt.subplots(figsize=(7, 3))
    plt.axis('off') 
    ax.imshow(image)
    st.pyplot(fig) 
    #st.write(pred)



    to_pick = []  
    for i in range(len(face_locations)):
        x1, x2, y1, y2 = X1[i], X2[i], Y1[i], Y2[i]
        to_pick.append(image_origin[y1:y2, x1:x2])

    img_num = st.slider("Which image?", 1, len(to_pick))
  
        
    if img_num:
        st.image(to_pick[img_num -1])
        st.write('1:', pred_values[img_num -1][0][0],':',  pred_values[img_num -1][0][1],'%') 
        st.write('2:', pred_values[img_num -1][1][0],':',  pred_values[img_num -1][1][1],'%')
        st.write('3:', pred_values[img_num -1][2][0],':',  pred_values[img_num -1][2][1],'%') 


    



HTML3 = f"""
<h2 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">Press START below to capture live emotions.</span></h2>
"""

st.write(HTML3, unsafe_allow_html=True)

class VideoTransformer(VideoTransformerBase):
    #inherit from streamlit_webrtc transformer
    def transform(self, frame):
        #convert mesh to array using bgr as this works best with cv2
        frame = frame.to_ndarray(format="bgr24")

        #locate all faces in the array image 
        face_locations = detector.detect_faces(frame)

        for face in range(len(face_locations)):
            #get location on box in the correct format for cv2
            x1, y1, width, height =face_locations[face]["box"]
            x2, y2 = x1 + width, y1 + height

            #select only the face from the image and predict emotions with the imported model
            cropped = frame[y1:y2, x1:x2]
            pred = model.predict(np.expand_dims(resize(cropped, [224, 224]),axis=0)/255.0)[0]

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
            #draw the box around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #retunr the image with all boxes and predictions on top 
        return frame


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS)