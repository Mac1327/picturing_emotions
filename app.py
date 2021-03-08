import streamlit as st
import matplotlib.pyplot as plt
from picturing_emotions.face_detector import DetectFace
import numpy as np
from tensorflow import keras
from tensorflow.image import resize
import cv2

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



if uploaded_file is not None:
    detector = DetectFace(uploaded_file)
    image = plt.imread(uploaded_file)
    image_zero = image.copy()
    faces, X1, X2, Y1, Y2 = detector.get_faces(save=True)
    pred_values = {}
    
    for i in range(len(faces)):
        pred = model.predict(np.expand_dims(resize(np.squeeze(faces[i]), [224, 224]),axis=0)/255.0)[0]
        top_values = pred.argsort()[-3:]
        prediction1 = classes[top_values[2]]
        pred1       = pred[top_values[2]]
        prediction2 = classes[top_values[1]]
        pred2       = pred[top_values[1]]
        prediction3 = classes[top_values[0]]
        pred3       = pred[top_values[0]]
     

        x1, x2, y1, y2 = X1[i], X2[i], Y1[i], Y2[i]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 30)
        cv2.putText(image, f"1: {prediction1}:{round(pred1*100)}%", (x1, y1-310), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (36,255,12), 10)
        cv2.putText(image, f"2: {prediction2}:{round(pred2*100)}%", (x1, y1-160), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (36,255,12), 10)
        cv2.putText(image, f"3: {prediction3}:{round(pred3*100)}%", (x1, y1-20), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (36,255,12), 10)

        pred_values[i] =  {prediction1: round(pred1*100),
                           prediction2: round(pred2*100),
                           prediction3: round(pred3*100)}

    fig, ax = plt.subplots(figsize=(7, 3))
    plt.axis('off') 
    ax.imshow(image)
    st.pyplot(fig) 
    #st.write(pred)


    to_pick = []  
    for i in range(len(faces)):
        x1, x2, y1, y2 = X1[i], X2[i], Y1[i], Y2[i]
        to_pick.append(image_zero[y1:y2, x1:x2])
     

    pick_img = st.sidebar.radio("Which image?", 
            [x for x in range(1, len(to_pick) +1)])

    #st.image(to_pick)

    # do something with what the user selected here
    if pick_img:

    #for i in to_pick:
        st.image(to_pick[pick_img -1])
        st.write(pred_values[pick_img -1])

