.PHONY: run clean test install_deps

run:
	@echo "Running main..."
	@python src/main.py
	@echo "Experiment completed."
clean:
	@echo "Cleaning up experiment runtime directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf experiment_results
	@rm -rf lib
	@rm -rf data/qdrant_data/*
	@rm -rf .pytest_cache
	@echo "Cleanup complete."
test:
	@echo "Running tests..."
	@pytest tests/
	@echo "All tests completed."
install_deps:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@echo "Dependencies installed."