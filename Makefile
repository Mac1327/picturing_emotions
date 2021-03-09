# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* picturing_emotions/*.py

black:
	@black scripts/* picturing_emotions/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr picturing_emotions-*.dist-info
	@rm -fr picturing_emotions.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#            GCP
# ----------------------------------
# path of the file to upload to gcp (the path of the file should be absolute or should match the directory where the make command is run)
DATA_PATH=raw_data/data_3000_compressed.zip
MODEL_PATH=raw_data/vg_face_model.zip

# project id
PROJECT_ID=picturingemotions

# bucket name
BUCKET_NAME=emotionsbucket

# bucket directory in which to store the uploaded file (we choose to name this data as a convention)
DATA_BUCKET_FOLDER=data
MODEL_BUCKETFOLDER=model

# name for the uploaded file inside the bucket folder (here we choose to keep the name of the uploaded file)
# BUCKET_FILE_NAME=another_file_name_if_I_so_desire.csv
DATA_BUCKET_FILE_NAME=$(shell basename ${DATA_PATH})
MODEL_BUCKET_FILE_NAME=$(shell basename ${MODEL_PATH})

REGION=europe-west1

set_project:
	-@gcloud config set project ${PROJECT_ID}

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	-@gsutil cp ${DATA_PATH} gs://${BUCKET_NAME}/${DATA_BUCKET_FOLDER}/${DATA_BUCKET_FILE_NAME}

upload_model:
	-@gsutil cp ${MODEL_PATH} gs://${BUCKET_NAME}/${MODEL_BUCKETFOLDER}/${MODEL_BUCKET_FILE_NAME}

# ----------------------------------
#            GCP Online Training
# ----------------------------------

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

REGION=europe-west1

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=picturing_emotions
FILENAME=predict

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=picturing_emotions_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs


# ----------------------------------
#            Uvicorn
# ----------------------------------
run_api:
	uvicorn api.api:app --reload

# ----------------------------------
#            Streamlit
# ----------------------------------

streamlit_app:
	-@streamlit run app.py  --server.port 8080

streamlit_image:
	-@streamlit run app_image.py  --server.port 8080
streamlit_combined:
	-@streamlit run app_combined.py  --server.port 8080

# ----------------------------------
#            Docker Image
# ----------------------------------

DOCKER_IMAGE_NAME=emotions03

docker_build:
	docker build -t eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} . 

docker_run:
	docker run -p 8080:8080 eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

docker_push:
	docker push eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME}

gc_deploy:
	gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region europe-west1 --cpu 4 --memory 4Gi