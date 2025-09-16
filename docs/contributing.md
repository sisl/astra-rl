# Contributing

!!! warning
    
    This documentation (and project) is a work in progress. If you have any questions or suggestions, please feel free to open an issue on the [GitHub repository](https://github.com/sisl/astra-rl)

Thank you for your interest in contributing to ASTRA-RL! We welcome contributions from the community to help improve the toolbox and its documentation.

## How to Contribute

**1) Fork the Repository**: Start by forking the [ASTRA-RL repository](https://github.com/sisl/astra-rl) on GitHub.

**2) Clone Your Fork**: Clone your forked repository to your local machine:
```bash
git clone https://github.com/YOUR_USERNAME/astra-rl.git
```

**3) Create a Branch**: Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

**4) Make Changes**: Make your changes to the code or documentation. Ensure that your code adheres to the project's coding standards and style guidelines.

**5) Run Tests**: Before committing your changes, run the tests to ensure everything is working correctly:
```bash
uv run pytest
```

If your tests require a GPU, you can run them with the `--gpu` option to enable GPU tests:
```bash
uv run pytest --gpu
```

**6) Commit Your Changes**: Commit your changes with a descriptive commit message:
```bash
git commit -m "Add feature: your-feature-name"
```

**7) Push Your Changes**: Push your changes to your forked repository:
```bash
git push origin feature/your-feature-name
```

**8) Create a Pull Request**: Go to the original repository on GitHub and create a pull request (PR) from your branch. Provide a clear description of the changes you made and any relevant context. Fill out the PR template to help us understand your changes better.

## Development

This section provides instructions for setting up the development environment and running tests.

To start, we _STRONGLY_ recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. This will ensure that you have the correct dependencies and versions installed.

### Setting Up the Development Environment

**Step 1**: Clone the repository:

```bash
git clone https://github.com/sisl/astra-rl.git
cd astra-rl
```

**Step 2**: Sync package dependencies:

```bash
uv sync --dev
```

This will create a `.venv` directory in the project root with all the necessary dependencies installed.

**Step 3**: Install pre-commit hooks:

```bash
uv run pre-commit install
```

This will ensure that the linter (`ruff`), formatter (`ruff`), and type checker (`mypy`) is happy with your code every time you commit.
   
### Running Tests

Assuming you've set up your environment using `uv`, you can run the tests using the following command:

```bash
pytest
```

or 

```bash
uv run pytest
```

To generate local coverage reports, you can use:

```bash
uv run coverage run -m pytest
uv run coverage report # Generate CLI report
uv run coverage html   # Generate HTML report
```

#### Running Tests with GPU

Some tests may require a GPU to run. You can enable GPU tests by passing the `--gpu` option:

```bash
uv run pytest --gpu
```

These tests will be _skipped_ by default unless you specify the `--gpu` option.

### Generating Documentation

To generate the documentation, you can use the following command:

```bash
uv run mkdocs serve
```

This will build the documentation and start a local server. You can then view the documentation in your web browser.
