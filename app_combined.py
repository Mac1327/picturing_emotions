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
#classes used for predcictions
classes = ['surprise',
            "fear",
            "disgust",
            "anger",
            "neutrality",
            "sadness",
            "happiness"]

#import model
model = keras.models.load_model("raw_data/vg_face_model")

#face detector for photos only 
detector = MTCNN()

#creat sidebar
st.sidebar.markdown(f"""
    # Picturing Emotions
    """)

#logo at top of page


image = Image.open('Logo.png')
st.image(image, caption='', use_column_width=False)


#set background colour
st.markdown("""
<style>
body {
    color: #000000;
    background-color: #f4d160;
}
</style>
    """, unsafe_allow_html=True)




#description text
HTML1 = f"""
<hr>
<h2 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">Let’s learn more about how our faces look when we are feeling different emotions. You can either upload a photo or  use your live webcam.</span></h2>
<hr>
<p><br></p>
"""
st.write(HTML1, unsafe_allow_html=True)


HTML2 = f"""
<h2 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">Please upload a photo to predict the emotions.</span></h2>
"""
st.write(HTML2, unsafe_allow_html=True)

#get user photo input
uploaded_file = st.file_uploader("Choose a photo")


if uploaded_file is not None:

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def upload_frame():

        image = plt.imread(uploaded_file)
        image = cv2.resize(image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
        image_origin = image.copy()

        face_locations = detector.detect_faces(image)

        X1, X2, Y1, Y2 = [],[],[],[]
        pred_values = {}

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
            

            cv2.rectangle(image, (x1, y1), (x2, y2), (105, 60, 114), 5)
            cv2.putText(image, f"1: {prediction1}:{round(pred1*100)}%", (x1, y2+30), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (244, 139, 41), 2)
            cv2.putText(image, f"2: {prediction2}:{round(pred2*100)}%", (x1, y2+60), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (105, 60, 114), 2)
            cv2.putText(image, f"3: {prediction3}:{round(pred3*100)}%", (x1, y2+90), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (244, 139, 41), 2)

            pred_values[i] =  [[prediction1, round(pred1*100)],
                            [prediction2, round(pred2*100)],
                            [prediction3, round(pred3*100)]]

            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        return image, image_origin, X1, X2, Y1, Y2, pred_values, face_locations

    image, image_origin, X1, X2, Y1, Y2, pred_values, face_locations = upload_frame()
    
    fig, ax = plt.subplots(figsize=(7, 3))
    plt.axis('off') 
    ax.imshow(image)
    st.pyplot(fig) 


    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def pick_img():
        to_pick = []  


        for i in range(len(face_locations)):
            x1, x2, y1, y2 = X1[i], X2[i], Y1[i], Y2[i]
            to_pick.append(image_origin[y1:y2, x1:x2])
        return to_pick

    to_pick = pick_img()

    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def emoji(cat):
        if cat == 'happiness':
            happy_img = Image.open('emoji/disgust.png')
            return happy_img
        elif cat == 'sadness':
            sad_img = Image.open('emoji/disgust.png')
            return sad_img
        elif cat == 'fear':
            scared_img = Image.open('emoji/disgust.png') 
            return scared_img
        elif cat == 'disgust':
            disgust_img = Image.open('emoji/disgust.png')
            return disgust_img
        elif cat == 'anger':
            angry_img = Image.open('emoji/disgust.png')
            return angry_img
        elif cat == 'surprise':
            surprise_img = Image.open('emoji/disgust.png')
            return surprise_img
        else:
            neutral_img = Image.open('emoji/disgust.png')
            return neutral_img 

    @st.cache(allow_output_mutation=True)
    def get_mutable():
        return {}


    if len(to_pick) > 1:
        img_num = st.slider("Which image?", 1, len(to_pick))
    else:
        img_num = 1

    caching_object = get_mutable()
    caching_object[img_num] = [to_pick[img_num -1], pred_values[img_num -1][0][0],  pred_values[img_num -1][0][1],
    pred_values[img_num -1][1][0], pred_values[img_num -1][1][1], pred_values[img_num -1][2][0],
    pred_values[img_num -1][2][1], emoji(pred_values[img_num -1][0][0])]

        
    if img_num:

        image1 = caching_object[img_num][0]
        image2 = caching_object[img_num][7]
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
        plt.axis('off')
        ax1.axis('off')
        fig.patch.set_facecolor('xkcd:mint green')
        ax1.imshow(image1)
        ax2.imshow(image2)
        st.pyplot(fig) 
     
        
        st.write('1:', caching_object[img_num][1],':', caching_object[img_num][2],'%') 
        st.write('2:',  caching_object[img_num][3],':', caching_object[img_num][4],'%')
        st.write('3:',  caching_object[img_num][5],':', caching_object[img_num][6],'%')   




HTML3 = f"""
<h2 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">Press START below to capture live emotions.</span></h2>
"""

# load the face detector xml
haar_cascade = cv2.CascadeClassifier("haar_face.xml")

st.write(HTML3, unsafe_allow_html=True)

class VideoTransformer(VideoTransformerBase):
    #inherit from streamlit_webrtc transformer
    def transform(self, frame):
        #convert mesh to array using bgr as this works best with cv2
        frame = frame.to_ndarray(format="bgr24")

        #locate all faces in the array image 
        faces_box = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=11)

        for (x1,y1,w,h) in faces_box:
            #creat box locations 
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (114, 60, 105), thickness=4)      
            
            #crop the face from the image and transform to correct model inpuit (1,224,244,3).
            # Must divide by 225.0
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
            cv2.putText(frame, f"1: {prediction1}:    {round(pred1*100)}%", (x1, y1-60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (41, 139, 244), 2)
            cv2.putText(frame, f"2: {prediction2}:    {round(pred2*100)}%", (x1, y1-35), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (114, 60, 105), 2)
            cv2.putText(frame, f"3: {prediction3}:    {round(pred3*100)}%", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (41, 139, 244), 2)
        #retunr the image with all boxes and predictions on top 
        return frame


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings=WEBRTC_CLIENT_SETTINGS)

HTML3 = f"""
<h2 style="text-align: left;"><span style="font-family: Helvetica; color: rgb(255, 255, 255);">You can pause the video when you want by pressing the pause button or by clicking the video.</span></h2>
"""
st.write(HTML3, unsafe_allow_html=True)