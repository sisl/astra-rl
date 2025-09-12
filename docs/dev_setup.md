# Development Setup Guide

If you want to contribute to ASTRA-RL, modify it for your needs, or run the examples from the repository, follow these steps.

### Prerequisites

We **strongly** recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. This ensures you have the correct dependencies and versions installed.

### Setting Up the Development Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sisl/astra-rl.git
   cd astra-rl
   ```

2. **Sync package dependencies:**

   ```bash
   uv sync --dev
   ```

   This creates a `.venv` directory in the project root with all necessary dependencies installed.
   
3. **Install pre-commit hooks:**

   ```bash
   uv run pre-commit install
   ```
   
   This ensures that the linter (`ruff`), formatter (`ruff`), and type checker (`mypy`) validate your code before each commit.

### Running Tests

After setting up your development environment, you can run tests using:

```bash
pytest
```

or with uv:

```bash
uv run pytest
```

### Generating Coverage Reports

To generate local coverage reports:

```bash
uv run coverage run -m pytest
uv run coverage report  # Generate CLI report
uv run coverage html    # Generate HTML report
```

The HTML report will be available in the `htmlcov` directory.

### Building Documentation Locally

To build and serve the documentation locally:

```bash
uv run mkdocs serve
```

This will start a local server at `http://127.0.0.1:8000` where you can preview documentation changes.

### Code Quality Tools

The project uses several tools to maintain code quality:

- **Ruff**: For linting and formatting
- **MyPy**: For type checking
- **Pre-commit**: For running checks before commits

To run these manually:

```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy .

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Troubleshooting

### Common Issues

1. **Import errors after installation:**
   - Ensure you're using the correct Python environment
   - Try reinstalling: `pip install --upgrade --force-reinstall astra-rl`

2. **Development environment issues:**
   - Make sure uv is properly installed
   - Try removing `.venv` and running `uv sync --dev` again

3. **GPU/CUDA issues:**
   - Ensure you have the appropriate PyTorch version for your CUDA installation
   - Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help

If you encounter issues:

- Check the [GitHub Issues](https://github.com/sisl/astra-rl/issues) for similar problems
- Open a new issue with details about your environment and the problem
- Join the discussions on the repository's Discussions tab