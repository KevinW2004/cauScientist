clean:
	@echo "Cleaning up experiment runtime directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf cma_experiments
	@rm -rf lib
	@rm -rf visualizations
	@echo "Cleanup complete."