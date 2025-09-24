# Installation Guide

This guide covers both basic installation for users and development setup for contributors.

## Basic Installation (For Users)

To use the ASTRA-RL toolbox in your projects, you can install it directly from PyPI:

```bash
pip install astra-rl
```

After installation, you can import the library in your Python code:

```python
import astra_rl
# or
import astra_rl as astral
```

### Optional Dependencies

If you plan to use Weights & Biases for experiment tracking, then install with
optional dependencies:

```bash
pip install "astra-rl[wandb]"
export WANDB_API_KEY=your_wandb_api_key_here
```

That's it! You should now be able to use ASTRA-RL in your projects.