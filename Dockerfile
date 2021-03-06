FROM python:3.8.6-buster

COPY api /api
COPY picturing_emotions /picturing_emotions
COPY raw_data/vg_face_model /raw_data/vg_face_model
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD streamlit run app.py  --server.port 8080