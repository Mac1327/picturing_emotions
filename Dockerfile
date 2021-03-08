FROM python:3.8.6-buster

#COPY app.py /app.py
COPY app_image.py /app_image.py

COPY picturing_emotions /picturing_emotions
COPY raw_data/vg_face_model /raw_data/vg_face_model
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install opencv-contrib-python

CMD streamlit run app_image.py  --server.port 8080