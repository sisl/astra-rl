# Adaptive Stress Testing for Robust AI & Reinforcement Learning (ASTRA-RL)

Welcome to the ASTRA-RL toolbox documentation! This documentation provides an overview of the ASTRA-RL toolbox, its features, and how to use it effectively.

## What is ASTRA-RL?

ASTRA-RL is a Python toolbox for **testing and evaluating** language models and generative AI systems. It provides a modular framework for using reinforcement learning to systematically discover how language models respond to adversarial or challenging inputs.

### What is Adversarial Testing for Language Models?

**Adversarial testing** is the process of stress-testing AI systems by trying to find inputs that cause them to behave in harmful or undesirable ways. Think of it like security testing for language models - you're trying to discover vulnerabilities before they're exploited in the real world.

ASTRA-RL automates this process using **reinforcement learning**. Instead of manually crafting adversarial prompts, you train an "auditor" model to automatically generate prompts that elicit unsafe responses from your target model. This is:

- **Faster** than manual testing once the auditor is trained
- **More systematic** at finding failure modes
- **Scalable** to large-scale evaluation

### Who Should Use ASTRA-RL?

This toolbox is designed for:

- **Researchers** studying AI safety and robustness
- **ML engineers** evaluating production language models
- **Safety teams** stress-testing conversational AI systems
- **Developers** building safer AI applications

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

## Key Features

- **Modular Architecture**: Easily swap components for your specific use case
- **Pre-built Algorithms**: Support for PPO, DPO, IPO out of the box
- **Multiple Scorers**: Integration with Llama-Guard 3, Detoxify, and custom scorers
- **HuggingFace Compatible**: Seamless integration with HuggingFace models
- **Extensible Framework**: Build custom systems, samplers, and solvers

## Support

If you encounter any issues or have questions:

- Check the [Tutorials](tutorials/index.md) for common use cases
- Review the [API documentation](api/index.md) for detailed information
- Open an issue on [GitHub](https://github.com/sisl/astra-rl/issues)
