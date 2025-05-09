---
license: mit
tags:
- finance
- market_simulation
paper:
- https://arxiv.org/abs/2409.07486
---
# Model Card for MarS Order Model

A cutting-edge financial market simulation engine powered by the Large Market Model (LMM), designed to generate realistic, interactive, and controllable **order sequences** for market simulation. More details can be found in the paper [MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model](https://arxiv.org/abs/2409.07486) and github repo [MarS](https://github.com/microsoft/MarS).

## Quick Start


### Installation Options

#### Option 1: Using VS Code Dev Containers (Recommended)

We provide a fully configured development environment using VS Code Dev Containers:

```bash
git clone https://github.com/microsoft/MarS.git
cd MarS
```

Then, with VS Code and the Dev Containers extension installed:
1. Open the project folder in VS Code
2. **Important**: Before reopening in container, modify the `.devcontainer/devcontainer.json` file to change `"source=/data/"` to `<your/data/path>` exists on your host machine
3. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Dev Containers: Reopen in Container"
4. The container will build with all dependencies and extensions configured
5. Once inside the container, install the project dependencies:
```bash
pip install -e .[dev]
```

#### Option 2: Using Docker Directly

```bash
git clone https://github.com/microsoft/MarS.git
cd MarS
docker build -t mars-env -f .devcontainer/Dockerfile .
# Modify this path to match your data directory
docker run -it --cap-add=SYS_ADMIN --device=/dev/fuse --security-opt=apparmor:unconfined --shm-size=20gb --gpus=all --privileged -v <your/data/path>:/data -v $(pwd):/workspaces/MarS -w /workspaces/MarS mars-env
# Inside the container
pip install -e .[dev]
```

> **Important**: We strongly recommend using docker to run MarS. Direct installation without Docker is not supported due to specific system dependencies and CUDA requirements.

### Download Model and Pre-requisites

We've simplified downloading all necessary components (model, converters, validation samples, and stylized facts data) using a single script:

```python
python download.py
```
> Note: The download requires sufficient disk space and may take some time depending on your internet connection.

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
