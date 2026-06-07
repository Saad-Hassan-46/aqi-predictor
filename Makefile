.PHONY: install test feature train app

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

feature:
	python pipelines/feature_pipeline.py

train:
	python pipelines/training_pipeline.py --lookback 7

app:
	streamlit run app/app.py
