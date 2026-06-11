.PHONY: generate-data compute-features train-data train serve monitor all docker-up docker-down

generate-data:
	python data/generate_data.py

compute-features:
	python feature_store/compute_features.py

train-data:
	python training/create_training_data.py

train:
	python training/train_models.py

serve:
	uvicorn serving.main:app --reload --port 8000

monitor:
	python monitoring/evidently_reports.py
	streamlit run monitoring/dashboard.py

all: generate-data compute-features train-data train monitor

docker-up:
	docker-compose -f docker/docker-compose.yml up --build

docker-down:
	docker-compose -f docker/docker-compose.yml down
