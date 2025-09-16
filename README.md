<!-- <p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/pages/assets/logo-gold.png" alt="Brahe"></a>
</p> -->
<p align="center">
    <em>ASTRA - Adaptive Stress Testing for Robust AI</em>
</p>
<p align="center">
<a href="https://github.com/sisl/astra-rl/actions/workflows/ci.yml" target="_blank">
    <img src="https://github.com/sisl/astra-rl/actions/workflows/ci.yml/badge.svg" alt="Test">
</a>
<a href='https://coveralls.io/github/sisl/astra-rl?branch=main'><img src='https://coveralls.io/repos/github/sisl/astra-rl/badge.svg?branch=main' alt='Coverage Status' /></a>
<a href="https://sisl.github.io/astra-rl/index.html" target="_blank">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs">
</a>
<a href="https://github.com/sisl/astra-rl/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg", alt="License">
</a>
</p>

----

# ASTRA-RL Toolbox

ASTRA-RL is a python toolbox for training and evaluating language models and generative AI systems that
use textual inputs. It provides a set of tools for training, evaluating, and analyzing
language models, with a focus on applying reinforcement learning based refinement techniques
to improve evaluator model performance.

## Installation

To install the ASTRA-RL toolbox, you can use pip. The package is available on PyPI, so you can install it directly from there.

```bash
pip install astra-rl
```

You can then import the library in your Python code:

```python
import astra_rl
# or
import astra_rl as astral
```

## Development

This section provides instructions for setting up the development environment and running tests.

To start, we _STRONGLY_ recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. This will ensure that you have the correct dependencies and versions installed.

### Setting Up the Development Environment

1. Clone the repository:

   ```bash
   git clone https://github.com/sisl/astra-rl.git
   cd astra-rl
   ```

2. Sync package dependencies:

   ```bash
   uv sync --dev
   ```

   This will create a `.venv` directory in the project root with all the necessary dependencies installed.
   
3. Install pre-commit hooks:

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
