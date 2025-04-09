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
# Clone the repository
git clone https://github.com/microsoft/MarS-Model-Release.git
cd MarS-Model-Release
# Install dependencies
pip install -e .[dev]
```

### Download and Load Model

The model is available on Hugging Face Hub. You can download and load it using:

```python
from market_simulation.models.order_model import OrderModel
from market_simulation.conf import C
# Load model directly from Hugging Face Hub
model = OrderModel.from_pretrained(C.model_serving.repo_id)
```

### Download and Configure Converters

The model requires converters to function properly. These converters are used to convert various market data (e.g., price, volume, intervals) into appropriate formats. The converters are available in the same Hugging Face repository as the model:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from market_simulation.conf import C

# Download the converters directory from the model repository
snapshot_download(
    repo_id=C.model_serving.repo_id,
    local_dir=C.directory.input_root_dir,
    allow_patterns=["converters/*"],
)

# These converter files will be automatically loaded from the default path when needed
# See `market_simulation/states/order_state.py` for details about the converters

# Initialize the converter if you need to use it in your own code
from market_simulation.states.order_state import Converter
converter_dir = Path(C.directory.input_root_dir) / C.order_model.converter_dir
converter = Converter(converter_dir=converter_dir)
```

### Starting the Order Model Ray Server

MarS uses [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to deploy the order model as a scalable, production-ready service. Ray Serve is a scalable model serving framework built on Ray that makes it easy to deploy ML models in production.

To start the order model Ray server:

```bash
# Run the start script
bash scripts/start-order-model.sh
```

This script will:

1. Stop any existing Ray cluster
2. Start a new Ray cluster with a head node
3. Deploy the order model service using the configuration in `ray_serving.yaml`
4. Display the status of the deployed service

Once the server is running, you can use the `ModelClient` to interact with it:

```python
from market_simulation.rollout.model_client import ModelClient
from market_simulation.conf import C

# Create a client to interact with the model server
client = ModelClient(
    model_name=C.model_serving.model_name,
    ip=C.model_serving.ip,
    port=C.model_serving.port,
)

# Get predictions from the model
predictions = client.get_prediction(your_input_data)
```

**Note:** Running the Ray server requires significant computational resources, especially if you're using GPU acceleration. Make sure your system meets the hardware requirements specified in the "Compute Infrastructure" section.

#### Testing the Ray Server

Once the Ray server is running, you can test it using the provided test functions:

1. **Interactive Demo**:

   ```bash
   # Run the Streamlit demo
   streamlit run market_simulation/examples/demo/home_app.py
   ```

   This will launch a web interface where you can:
   - Generate Market Forecasts
   - Analyze Market Impact

   The Interactive Demo provides an intuitive way to interact with the model through a user-friendly interface, making it ideal for quick exploration and visualization of the model's capabilities.

2. **Run a Market Forecast Simulation**:

   ```bash
   # Run the forecast script
   python -m market_simulation.examples.forecast
   ```

   This script performs a more comprehensive test of the Ray server by:
   - Creating a market simulation environment with multiple agents
   - Using the ModelClient to communicate with the Ray server for order generation
   - Running multiple simulation rollouts with different random seeds
   - Visualizing the resulting price trajectories and saving them as images

   The script will:
   - Initialize a noise agent to establish initial market conditions
   - Use a background agent that relies on the Ray server for order generation
   - Run multiple simulations in parallel (if not in debug mode)
   - Save the simulation results to compressed files
   - Generate visualizations of the price trajectories

   The output will be saved in the `output/forecasting-example` directory, with files named `rollouts-seed{seed}-run{run}.zstd` for the simulation data and `rollouts-seed{seed}-run{run}.png` for the visualizations.

   This approach offers more flexibility and control over the simulation parameters, allowing for more advanced testing and analysis of the model's performance.

**Prerequisites for Testing:**
- The Ray server must be running and accessible at the configured IP and port
- Validation samples must be available in the expected location
- The model should be configured with minimal temperature for deterministic outputs

### Run Stylized Facts Analysis

To run the stylized facts analysis, you need to download the stylized facts data file:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from market_simulation.conf import C

# Download the stylized facts file from the model repository
snapshot_download(
    repo_id=C.model_serving.repo_id,
    local_dir=C.directory.input_root_dir,
    allow_patterns=["stylized-facts/*"],
)

# Run the stylized facts analysis script
python market_simulation/examples/report_stylized_facts.py
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
