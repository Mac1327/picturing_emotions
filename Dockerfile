FROM python:3.8.6-buster

COPY app.py /app.py
COPY picturing_emotions /picturing_emotions
COPY raw_data/vg_face_model /raw_data/vg_face_model
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN python -m pip install opencv-contrib-python

CMD streamlit run app.py  --server.port 8080