clean:
	@echo "Cleaning up __pycache__ directories..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +