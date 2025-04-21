---
license: mit
tags:
- finance
- market_simulation
paper:
- https://arxiv.org/abs/2409.07486
---
# Model Card for MarS Order Model

A cutting-edge financial market simulation engine powered by the Large Market Model (LMM), designed to generate realistic, interactive, and controllable order sequences for market simulation.

## Quick Start

### Installation

```bash
git clone https://github.com/microsoft/MarS.git
cd MarS
pip install -e .[dev]
```
### Download and Configure Converters

The model requires converters to function properly. These converters are used to convert various market data (e.g., price, volume, intervals) into appropriate formats. The converters are available in the same Hugging Face repository as the model:

```python
from huggingface_hub import snapshot_download
from market_simulation.conf import C

snapshot_download(
    repo_id=C.model_serving.repo_id,
    local_dir=C.directory.input_root_dir,
    allow_patterns=["converters/*"],
)
```
> These converter files will be automatically loaded from the default path when needed.
See `market_simulation/states/order_state.py` for details about the converters


### Download and Load Model

The model is available on Hugging Face Hub. You can download and load it using:

```python
from market_simulation.models.order_model import OrderModel
from market_simulation.conf import C
model = OrderModel.from_pretrained(C.model_serving.repo_id)
```


### Starting the Order Model Ray Server

MarS uses [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to deploy the order model as a scalable, production-ready service.
To start the order model Ray server:

```bash
bash scripts/start-order-model.sh
```

> **Prerequisites:**
> - The Ray server must be running and accessible at the configured IP and port
> - Sufficient computational resources are required to run the model

To explore all of our demos in a user-friendly interface:

```bash
streamlit run market_simulation/examples/demo/home_app.py
```

The demo applications are designed to provide a quick and visual understanding of each tool's capabilities. However, there are some important considerations:

> **Using Demos vs Scripts**:
> - If you want to quickly understand what these tools can do, run the Streamlit demos for an interactive experience.
> - If you need to use these tools with your own data or in production, you'll need to modify the corresponding scripts (`report_stylized_facts.py`, `forecast.py`, `market_impact.py`) directly.

### Direct Model Interaction

If you want to interact with the model directly after starting the server, you can use the `ModelClient`.

```python
from market_simulation.rollout.model_client import ModelClient
from market_simulation.conf import C

client = ModelClient(
    model_name=C.model_serving.model_name,
    ip=C.model_serving.ip,
    port=C.model_serving.port,
)

predictions = client.get_prediction(your_input_data)
```

We also provide validation data on for testing purposes. You can download this data by running the following code:

```python
from huggingface_hub import snapshot_download
from market_simulation.conf import C

snapshot_download(
    repo_id=C.model_serving.repo_id,
    local_dir=C.directory.input_root_dir,
    allow_patterns=["validation-samples/*"],
)
```

## Model Details

### Model Description

MarS (Market Simulation) is a generative foundation model specifically designed for financial market simulation. It addresses the critical need for realistic, interactive, and controllable order generation in market simulations.

- **Developed by:** Microsoft Research
- **Model type:** Large Market Model (LMMM) for Financial Market Simulation
- **Language(s):** Python
- **License:** MIT

### Model Sources

- **Repository:** [MarS-Model-Release](https://github.com/microsoft/MarS-Model-Release)
- **Paper:** [MarS: A Financial Market Simulation Engine Powered by Generative Foundation Model](https://arxiv.org/abs/2409.07486)
- **Demo:** [Project Website](https://mars-lmm.github.io/)

## Uses

### Direct Use

The model is designed to generate realistic order sequences for market simulation, including:
- Order type (buy/sell/cancel)
- Price levels
- Order volumes
- Order time intervals

### Downstream Use

The model can be used for various market analysis applications:
1. Stylized Facts Report: Evaluating 11 key market characteristics
2. Market Forecast: Predicting future market prices and trends
3. Market Impact Analysis: Assessing the market impact of trading strategies

### Out-of-Scope Use

The model should not be used for:
- Direct trading decisions
- Financial advice
- Risk assessment without proper validation
- High-frequency trading without additional optimization

## Bias, Risks, and Limitations

### Technical Limitations
- Requires significant computational resources (128 GPUs for research simulations)
- Current implementation prioritizes validation over production optimization
- Fixed-length sequences with sliding windows limit KV-cache optimization

### Recommendations

Users should:
- Validate model outputs against historical data
- Consider computational requirements for production deployment
- Implement proper risk management measures
- Use the model as part of a comprehensive market analysis toolkit

## Training Details

### Training Data

The model is trained on order-level market data from the Chinese market, focusing on the top 500 liquid stocks.

### Training Procedure

#### Training Hyperparameters
- **Training regime:** Mixed precision training
- **Model architecture:** LLaMA-based transformer
- **Embedding dimension:** 1024
- **Number of layers:** 24
- **Number of attention heads:** 16
- **Dropout:** 0.1

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
- Order-level market data
- Market microstructure data
- Trading volume and price data

#### Factors
- Market conditions
- Trading volume
- Price levels
- Order types

#### Metrics
- Order generation accuracy
- Market microstructure realism
- Price impact prediction
- Volume prediction

### Results

The model successfully reproduces 9 out of 11 key market stylized facts, demonstrating strong performance in capturing market dynamics.

## Technical Specifications

### Model Architecture and Objective

The model uses a LLaMA-based transformer architecture with:
- Order tokenization layer
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

### Compute Infrastructure

#### Hardware Requirements
- GPU: NVIDIA GPUs with at least 16GB VRAM
- CPU: Multi-core processor
- RAM: 32GB minimum

#### Software Requirements
- Python 3.8 or 3.9
- PyTorch
- Hugging Face Transformers
- CUDA toolkit

## Citation

**BibTeX:**
```bibtex
@article{li2024mars,
  title={MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model},
  author={Li, Junjie and Liu, Yang and Liu, Weiqing and Fang, Shikai and Wang, Lewen and Xu, Chang and Bian, Jiang},
  journal={arXiv preprint arXiv:2409.07486},
  url={https://arxiv.org/abs/2409.07486},
  year={2024}
}
```
## Model Card Contact

For questions and feedback about this model, please contact:
- GitHub Issues: [MarS-Model-Release](https://github.com/microsoft/MarS-Model-Release/issues)
- Project Website: [mars-lmm.github.io](https://mars-lmm.github.io/)
- WeChat Group: Available in project documentation
- Discord: [Join our community](https://discord.gg/jW8gKDDEqS)
