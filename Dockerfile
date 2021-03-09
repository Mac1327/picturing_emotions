FROM python:3.8.6-buster

#COPY app.py /app.py
COPY app_combined.py /app_combined.py

COPY picturing_emotions /picturing_emotions
COPY raw_data/vg_face_model /raw_data/vg_face_model
COPY requirements.txt /requirements.txt
COPY 02.png /02.png
COPY haar_face.xml /haar_face.xml


RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install opencv-contrib-python

CMD streamlit run app_combined.py  --server.port 8080