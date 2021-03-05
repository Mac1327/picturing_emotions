from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from picturing_emotions import predict

MODEL_LOCATITON = "raw_data/vg_face_model"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict_fare/?key=2012-10-06 12:10:20.0000001&pickup_datetime=2012-10-06 12:10:20 UTC&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


@app.get("/")
def index():
    return {"ok": "HELLO!"}

@app.get("/predict_faces/")
def predict_faces(input_list):

    list_of_arrays_of_images = input_list

    predictions = predict(list_of_arrays_of_images, MODEL_LOCATITON)

    dict_of_predictions = {}

    for face_number in len(range(predictions)):
        dict_of_predictions[face_number] = predictions[face_number]
    return "dict_of_predictions"