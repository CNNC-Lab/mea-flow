.PHONY: help install install-dev install-full test test-verbose test-fast test-coverage lint format docs docs-serve clean build upload pre-commit verify

# Default target
help: ## Show this help message
	@echo "MEA-Flow Development Commands"
	@echo "============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation targets
install: ## Install package in editable mode
	uv sync

install-dev: ## Install with development dependencies
	uv sync --group dev

install-full: ## Install with all optional dependencies
	uv sync --all-extras --group dev

# Testing targets
test: ## Run tests
	uv run pytest tests/ -v

test-verbose: ## Run tests with verbose output
	uv run pytest tests/ -v --tb=long

test-fast: ## Run tests excluding slow tests
	uv run pytest tests/ -v -m "not slow"

test-coverage: ## Run tests with coverage report
	uv run pytest tests/ --cov=mea_flow --cov-report=html --cov-report=term --cov-report=xml

test-unit: ## Run only unit tests
	uv run pytest tests/ -v -m "unit"

test-integration: ## Run only integration tests
	uv run pytest tests/ -v -m "integration"

# Code quality targets
lint: ## Run all linting checks
	uv run black --check --diff src/ tests/
	uv run flake8 src/ tests/
	uv run mypy src/mea_flow --ignore-missing-imports

format: ## Format code with black and isort
	uv run black src/ tests/
	uv run isort src/ tests/

format-check: ## Check if code is properly formatted
	uv run black --check src/ tests/
	uv run isort --check-only src/ tests/

security: ## Run security checks
	uv run safety check --json || true
	uv run bandit -r src/ -f json || true

# Documentation targets
docs: ## Build documentation
	cd docs/ && uv run sphinx-build -b html . _build/html

docs-clean: ## Clean documentation build
	cd docs/ && rm -rf _build/

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-linkcheck: ## Check documentation links
	cd docs/ && uv run sphinx-build -b linkcheck . _build/linkcheck

# Pre-commit targets
pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

# Build and release targets
clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	uv build

build-check: build ## Build and check package
	uv run twine check dist/*

upload-test: build-check ## Upload to TestPyPI
	uv run twine upload --repository testpypi dist/*

upload: build-check ## Upload to PyPI
	uv run twine upload dist/*

# Verification and setup targets
verify: ## Run installation verification
	uv run python verify_install.py

setup-dev: install-dev pre-commit-install ## Complete development setup
	@echo "Development environment set up successfully!"
	@echo "Run 'make verify' to test the installation"

# Notebook targets
notebook: ## Start Jupyter Lab
	uv run jupyter lab

notebook-test: ## Test that notebooks can be executed
	@for notebook in notebooks/*.ipynb; do \
		if [ -f "$$notebook" ]; then \
			echo "Testing notebook: $$notebook"; \
			uv run jupyter nbconvert --to notebook --execute --inplace "$$notebook"; \
		fi; \
	done

# CI simulation targets
ci-test: ## Simulate CI test pipeline
	make lint
	make test-coverage
	make docs
	make build-check

ci-quick: ## Quick CI check (fast tests only)
	make format-check
	make test-fast
	make verify

# Docker targets
docker-build: ## Build Docker image
	docker build -t mea-flow .

docker-test: ## Run tests in Docker
	docker run --rm mea-flow pytest tests/ -v

docker-notebook: ## Run Jupyter in Docker
	docker run --rm -p 8888:8888 mea-flow jupyter lab --ip=0.0.0.0 --allow-root

# Benchmarking and profiling
benchmark: ## Run performance benchmarks
	uv run pytest tests/ -v -m "slow" --benchmark-only

profile: ## Profile test execution
	uv run pytest tests/test_analysis.py::TestAnalysisPerformance::test_large_dataset_performance --profile

# Release preparation
prepare-release: ## Prepare for release (run before tagging)
	make ci-test
	make docs
	@echo "Release preparation complete!"
	@echo "Ready to tag and release"

# Development helpers
deps-update: ## Update dependencies
	uv lock --upgrade

deps-check: ## Check for outdated dependencies  
	uv tree

env-info: ## Show environment information
	@echo "Python version:"
	@python --version
	@echo "uv version:"
	@uv --version
	@echo "Package version:"
	@python -c "import mea_flow; print(mea_flow.__version__)" 2>/dev/null || echo "Package not installed"

# Quick development workflow
dev: ## Quick development check (format + test-fast + verify)
	make format
	make test-fast
	make verify
	@echo "Development check complete!"