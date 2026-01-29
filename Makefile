.PHONY: run clean clean-db clean-all test inject_knowledge install_deps

info := @echo "[Makefile]"

run:
	$(info) "Running main..."
	@python src/main.py
	$(info) "Experiment completed."

clean:
	$(info) "Cleaning up experiment runtime directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf experiment_results
	@rm -rf lib
	@rm -rf .pytest_cache
	$(info) "Runtime temporary files cleanup complete."

clean-db:
	$(info) "Cleaning up vector database data..."
	@rm -rf data/qdrant_data/*
	$(info) "Vector database cleanup complete."

clean-all: clean clean-db
	$(info) "All cleanup complete."

test:
	$(info) "Running tests..."
	@pytest tests/
	$(info) "All tests completed."

inject_knowledge:
	$(info) "Injecting knowledge from all files in data/documents into vector database..."
	@python scripts/inject_knowledge.py
	$(info) "Knowledge injection complete."

install_deps:
	$(info) "Installing dependencies..."
	@pip install -r requirements.txt
	$(info) "Dependencies installed."