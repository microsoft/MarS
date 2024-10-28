# MarS: A Financial Market Simulation Engine Powered by Generative Foundation Model

## Introduction

MarS is a cutting-edge financial market simulation engine powered by the Large Market Model (LMM), a generative foundation model. MarS addresses the need for realistic, interactive, and controllable order generation. This paper's primary goals are to evaluate the LMM's scaling law in financial markets, assess MarS’s realism, balance controlled generation with market impact, and demonstrate MarS’s potential applications.

Below is a high-level overview diagram illustrating the core components, workflow, and potential applications of the MarS simulation engine:

<img src="doc/img/high-level-overview.png" alt="High-Level Overview of MarS" />

### Main Contributions
- We take the first step toward building a generative foundation model as a world model for financial market and verify the scaling law of the Large Market Model. It demonstrates the huge potential of this new direction of domain-specific foundation models.
- We design a realistic Market Simulation based on the LMM to fulfill two key requirements: generating target scenarios and modeling order market impacts, thereby unlocking LMM’s potential for meaningful applications.
- We showcase four types of downstream applications of MarS, demonstrating the significant potential of the MarS-based paradigm for the industry.

For more detailed information, please refer to our [paper](https://arxiv.org/abs/2409.07486) and [website](https://mars-lmm.github.io/).

## Current Release

We are excited to release the MarS simulation engine along with examples demonstrating its capabilities for market simulation. This release includes:
- [mlib](mlib): The core engine for generating and simulating financial market orders.
- [market_simulation](market_simulation): Example scripts illustrating how to use the MarS engine for market simulations.

The release of the pretrained model is currently undergoing internal review. We will make the model public once it passes the review. We look forward to sharing more features, examples, and applications in the future. Stay tuned for updates!


## Installation

The code is tested with Python 3.8 & 3.9. Run the following command to install the necessary dependencies:

```bash
pip install -e .[dev]
```

## Market Simulatin Library

**mlib** is a comprehensive library dedicated to market simulation, designed to be user-friendly, allowing users to focus on the design of states and agents.

Behind the scenes, we automatically:

- Refresh the orderbook with incoming orders.
- Update states with pertinent trade information.
- Distribute states and actions considering network and computational latency.

### Overall architecture

<img src="doc/img/mlib-flow.png" alt="mlib-architecture" />

#### Env

Env is a [gym](https://www.gymlibrary.dev/)-like interface. Below is an example of how to generate orders using env and a noise agent:

```python
agent = NoiseAgent(
    symbol=symbol,
    init_price=100000,
    interval_seconds=1,
    start_time=start_time,
    end_time=end_time,
)
env = Env(exchange, description="Noise agent simulation")
env.register_agent(agent)
env.push_events(create_exchange_events(config))
for observation in env.env():
    action = observation.agent.get_action(observation)
    env.step(action)
```

#### States

States are information available to agents, automatically updated with every trade information, including orders, transactions, and orderbook snapshots as defined in [trade_info.py](mlib/core/trade_info.py).

States are shared by agents and zero-copy during their lifetime, even in environments supporting delayed states.

- Creating a new state is straightforward. Here is an example of creating one that includes all transactions:

```python

class TransState(State):
    def __init__(self) -> None:
        super().__init__()
        self.transactons: List[Transaction] = []

    def on_trading(self, trade_info: TradeInfo):
        super().on_trading(trade_info)
        self.transactons.extend(trade_info.transactions)

    def on_open(self, cancel_transactions: List[Transaction], lob_snapshot: LobSnapshot, match_trans: Optional[Transaction] = None):
        super().on_open(cancel_transactions=cancel_transactions, lob_snapshot=lob_snapshot, match_trans=match_trans)
        self.transactons.extend(cancel_transactions)
        if match_trans:
            self.transactons.append(match_trans)

    def on_close(self, close_orderbook: Orderbook, lob_snapshot: LobSnapshot, match_trans: Optional[Transaction] = None):
        super().on_close(match_trans=match_trans, close_orderbook=close_orderbook, lob_snapshot=lob_snapshot)
        if match_trans:
            self.transactons.append(match_trans)
```

Once a new state is defined and registered with `exchange.register_state(state)`, it will be available when the agent wakes up.

So far, we have defined the following states:

- [trans_state](market_simulation/states/trans_state.py) contains all transactions.

- [trade_info_state](market_simulation/states/trade_info_state.py) contains all trade information.

#### Example: Run Simulation with Noise Agent

You can run the [run_simulaton.py](market_simulation/examples/run_simulation.py) for a complete example to perform market simulation with a noise agent.

```python
python market_simulation/examples/run_simulation.py
```

You can see the price trajectory generated from matching orders by the noise agent as follow:

![Noise Agent Simulation](doc/img/price_curves.png)

Note: This example demonstrates the use of MarS to simulate a market with a noise agent. For realistic market simulations, a more comprehensive model, such as the Large Market Model (LMM) in MarS, is typically required.


## Disclaimer

Users of the market simulation engine and the code should prepare their own agents which may be included trained models built with users’ own data, independently assess and test the risks of the model in a specify use scenario, ensure the responsible use of AI technology, including but limited to developing and integrating risk mitigation measures, and comply with all applicable laws and regulations. The market simulation engine does not provide financial opinions, nor is it designed to replace the role of qualified financial professionals in formulating, assessing, and approving finance products. The outputs of the market simulation engine do not reflect the opinions of Microsoft.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
