run:
	@echo "Running main..."
	@python src/main.py
	@echo "Experiment completed."
clean:
	@echo "Cleaning up experiment runtime directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf experiment_results
	@rm -rf lib
	@echo "Cleanup complete."