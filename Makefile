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
	@echo "make ingest: added in Phase 2 (LlamaIndex knowledge-base ingestion)"

eval:
	@echo "make eval: added in Phase 6 (golden Q&A evaluation)"
