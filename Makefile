.PHONY: up down test ingest eval run

up:
	docker compose up -d

down:
	docker compose down

run:
	uvicorn app.main:app --reload

test:
	pytest

ingest:
	python -m app.knowledge.ingest

eval:
	@echo "make eval: added in Phase 6 (golden Q&A evaluation)"
