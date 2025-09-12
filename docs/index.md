# Adaptive Stress Testing for Robust AI & Reinforcement Learning (ASTRA-RL)

Welcome to the ASTRA-RL toolbox documentation! This documentation provides an overview of the ASTRA-RL toolbox, its features, and how to use it effectively.

## What is ASTRA-RL?

ASTRA-RL is a Python toolbox for training and evaluating language models and generative AI systems that use textual inputs. It provides a set of tools for training, evaluating, and analyzing language models, with a focus on applying reinforcement learning based refinement techniques to improve evaluator model performance.

The toolbox is particularly designed for **LM red-teaming** - a process that helps identify and benchmark prompts that elicit harmful or otherwise undesirable behavior from target language models. This helps surface vulnerabilities and guides fine-tuning to reduce harmful outputs.

## Getting Started

### Quick Installation

To get started quickly with ASTRA-RL:

```bash
pip install astra-rl
```

Then import it in your Python code:

```python
import astra_rl
```

For detailed installation instructions, including development setup, see the **[Installation Guide](installation.md)**.

### Quick Links

- **[Installation Guide](installation.md)** - Detailed installation and setup instructions
- **[Tutorials](tutorials/index.md)** - Step-by-step guides for common tasks
- **[API Reference](api/index.md)** - Detailed documentation of all classes and functions

## Key Features

- **Modular Architecture**: Easily swap components for your specific use case
- **Pre-built Algorithms**: Support for PPO, DPO, IPO out of the box
- **Multiple Moderators**: Integration with Llama-Guard 3, Detoxify, and custom moderators
- **HuggingFace Compatible**: Seamless integration with HuggingFace models
- **Extensible Framework**: Build custom problems, environments, and solvers

## Documentation Structure

- **[Installation](installation.md)** - Setup instructions for users and developers
- **[Tutorials](tutorials/index.md)** - Learn how to use ASTRA-RL with hands-on examples
    - [Quick Start Training](tutorials/quick_start_training.md) - Train your first red-teaming model
    - [Quick Start Evaluation](tutorials/quick_start_evaluation.md) - Evaluate models with pre-trained attackers
    - [Customization Guides](tutorials/index.md#package-overview) - Adapt ASTRA-RL to your needs
- **[API Reference](api/index.md)** - Complete API documentation

## Support

If you encounter any issues or have questions:

- Check the [Tutorials](tutorials/index.md) for common use cases
- Review the [API documentation](api/index.md) for detailed information
- Open an issue on [GitHub](https://github.com/sisl/astra-rl/issues)
