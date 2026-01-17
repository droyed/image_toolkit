# Contributing to image_toolkit

First off, thank you for considering contributing to `image_toolkit`! It's people like you that make this tool faster, better, and more useful for the community.

This guide details how to set up your development environment, our coding standards, and the process for submitting pull requests.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Development Workflow](#development-workflow)
  - [Code Style](#code-style)
  - [Linting and Formatting](#linting-and-formatting)
- [Testing](#testing)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- `libjpeg-turbo` (required for PyTurboJPEG)

### Installation

1. **Fork and Clone** the repository:
   ```bash
   git clone https://github.com/yourusername/image_toolkit.git
   cd image_toolkit
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   We use a `Makefile` to simplify setup. To install the package in editable mode along with all development dependencies:
   ```bash
   make install
   ```
   Alternatively, using pip directly:
   ```bash
   pip install -e ".[dev]"
   ```

## Project Structure

The project follows the modern `src/` layout:

```
image_toolkit/
├── src/
│   └── image_toolkit/      # Main package code
│       ├── __init__.py
│       ├── handler.py      # ImageHandler class
│       ├── batch_handler.py  # BatchImageHandler class
│       ├── py.typed        # PEP 561 type marker
│       ├── core/           # Core functionality modules
│       │   ├── __init__.py
│       │   ├── io.py       # Image I/O operations
│       │   ├── transforms.py  # Image transformations
│       │   ├── analysis.py    # Image analysis
│       │   ├── annotations.py # Drawing/annotation tools
│       │   └── context.py     # Context managers
│       └── batch/          # Batch processing modules
│           ├── __init__.py
│           ├── operations.py     # Batch operations
│           ├── parallel.py       # Parallel processing
│           └── duplicate_analysis.py  # Duplicate detection
├── tests/                  # Test suite
│   ├── test_*.py          # Test modules
│   └── README.md          # Test documentation
├── docs/                   # Documentation
│   ├── guide/             # User guides
│   ├── api/               # API reference
│   └── advanced/          # Advanced topics
├── assets/                 # Example images and resources
├── pyproject.toml         # Project configuration
├── MANIFEST.in            # Distribution manifest
├── Makefile               # Development commands
└── README.md              # Project overview
```

The `src/` layout ensures that:
- Editable installs always import from the installed package, not local source
- Tests run against the installed package, improving reliability
- Package namespace is clearly separated from development files

## Development Workflow

1. **Create a Branch**: Always create a new branch for your feature or fix.
   ```bash
   git checkout -b feature/my-new-feature
   ```

### Code Style

We follow strict coding standards to keep the codebase clean and consistent.

- **Formatter**: [Black](https://github.com/psf/black)
- **Import Sorter**: [isort](https://github.com/PyCQA/isort)
- **Linter**: [Flake8](https://github.com/PyCQA/flake8)
- **Type Checker**: [mypy](https://mypy-lang.org/)

### Linting and Formatting

Before submitting your code, ensure it meets our style guidelines. We provide `make` commands for convenience:

- **Format Code** (applies Black and isort):
  ```bash
  make format
  ```

- **Run Linters** (checks with Flake8 and Mypy):
  ```bash
  make lint
  ```

Ensure `make lint` passes without errors before pushing your changes.

## Testing

- **Run All Tests**:
  ```bash
  make test
  # OR
  python run_all_tests.py
  ```

- **Run Specific Tests**:
  ```bash
  pytest tests/test_specific_file.py
  ```

- **Benchmarks**:
  If you are making performance-critical changes, consider running benchmarks:
  ```bash
  python tests/benchmark_parallel.py
  ```

## Submitting a Pull Request

1. **Update Documentation**: If you changed APIs or added features, update `README.md` and relevant files in `docs/`.
2. **Run Tests**: Ensure all tests pass.
3. **Check Style**: Run `make format` and `make lint`.
4. **Push Changes**: Push your branch to your fork.
5. **Open PR**: Create a Pull Request against the `main` branch.
   - Provide a clear title and description.
   - Reference any related issues (e.g., "Fixes #123").

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. Include:
- Steps to reproduce the bug.
- Expected vs. actual behavior.
- Your OS and Python version.

---
Happy Coding! 🚀
