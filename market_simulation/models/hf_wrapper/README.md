# Model Card for MarS Order Model

A cutting-edge financial market simulation engine powered by the Large Market Model (LMM), designed to generate realistic, interactive, and controllable order sequences for market simulation.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/microsoft/MarS-Model-Release.git
cd MarS-Model-Release

# Install dependencies
pip install -e .[dev]
```

### Download Model

The model is available on Hugging Face Hub. You can download it using:

```python
from huggingface_hub import hf_hub_download
from market_simulation.models.hf_wrapper.upload_model import MarsOrderModel

# Download model files
model_path = hf_hub_download(
    repo_id="microsoft/mars-order-model",
    filename="pytorch_model.bin",
    token="<your_hf_token>"  # Required for downloading
)

# Download tokenizer config
tokenizer_config_path = hf_hub_download(
    repo_id="microsoft/mars-order-model",
    filename="tokenizer_config.json",
    token="<your_hf_token>"
)
```

### Basic Usage

```python
import torch
from market_simulation.models.hf_wrapper.upload_model import MarsOrderModel

# Initialize model
model = MarsOrderModel(
    emb_dim=1024,
    num_layers=24,
    num_heads=16,
    num_bins_price_level=32,
    num_bins_pred_order_volume=32,
    num_bins_order_interval=16,
    num_max_orders=200,
    dropout=0.1,
)

# Load model weights
model.load_state_dict(torch.load(model_path))
model.eval()

# Generate orders
with torch.no_grad():
    # Your input features should be a tensor of shape [batch_size, num_max_orders * 15]
    # where 15 is the dimension of order features
    features = torch.randn(1, 200 * 15)  # Example input
    logits = model(features)

    # Sample orders
    orders = model.sample(features, temperature=1.0)
```

### Interactive Demo

For a quick start with the interactive demo:

```bash
# Run the Streamlit demo
streamlit run market_simulation/examples/demo/home_app.py
```

This will launch a web interface where you can:
1. View Stylized Facts Report
2. Generate Market Forecasts
3. Analyze Market Impact

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
  year={2024}
}
```

## Model Card Contact

For questions and feedback about this model, please contact:
- GitHub Issues: [MarS-Model-Release](https://github.com/microsoft/MarS-Model-Release/issues)
- Project Website: [mars-lmm.github.io](https://mars-lmm.github.io/)
- WeChat Group: Available in project documentation
- Discord: [Join our community](https://discord.gg/jW8gKDDEqS)
