.PHONY: up down test ingest eval run worker seed

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
	@echo "make eval: added in Phase 6 (golden Q&A evaluation)"
