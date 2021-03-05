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
        
    return dict_of_predictions