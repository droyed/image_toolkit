.PHONY: install test lint format clean

install:
	pip install -e .[dev]

test:
	python run_all_tests.py

lint:
	flake8 src/image_toolkit tests
	mypy src/image_toolkit

format:
	black src/image_toolkit tests
	isort src/image_toolkit tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov dist build
	rm -rf test_outputs/
	rm -rf logs/
