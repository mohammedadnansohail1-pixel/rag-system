.PHONY: help install test lint run docker-up docker-down docker-logs clean

help:
	@echo "RAG System Commands"
	@echo "==================="
	@echo "install      - Install dependencies"
	@echo "test         - Run all tests"
	@echo "lint         - Run linter"
	@echo "run          - Start API locally"
	@echo "demo         - Run interactive demo"
	@echo "docker-up    - Start all services (GPU)"
	@echo "docker-up-cpu - Start all services (CPU)"
	@echo "docker-down  - Stop all services"
	@echo "docker-logs  - View logs"
	@echo "clean        - Clean up caches"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/unit/ -v

lint:
	ruff check src/

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

demo:
	python scripts/demo_production.py

docker-up:
	docker-compose up -d --build

docker-up-cpu:
	docker-compose -f docker-compose.cpu.yml up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
