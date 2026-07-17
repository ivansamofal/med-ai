.PHONY: up down test ingest eval run worker seed sample-documents ab-eval ragas-eval

up:
	docker compose up -d

down:
	docker compose down

run:
	uvicorn app.main:app --reload

worker:
	python -m app.workers.recommendation_worker

test:
	pytest

ingest:
	python -m app.knowledge.ingest

seed:
	python -m scripts.seed_demo_data

eval:
	python -m app.eval.run_eval

sample-documents:
	python -m scripts.generate_sample_documents

ab-eval:
	python -m app.eval.run_experiment

ragas-eval:
	python -m app.eval.run_ragas_eval
