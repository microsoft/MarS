from pydantic import BaseModel
from pydantic_settings import BaseSettings

ENV_NESTED_DELIMITER = "__"


class ModelServing(BaseModel):
    """Config for model serving."""

    model_path: str = "order-model/20240423-0939-model-scaling-llama2-order-time-mid-rel-LOB-24-1024-16-0.0004.6.904300212860107.ckpt"
    model_name: str = "order-model"
    temperature: float = 1.0
    ip: str = "localhost"
    port: int = 8000

    num_gpus: float = 0.3  # 30% of GPU
    num_cpus: float = 0.2  # 20% of CPU
    max_batch_size: int = 16

    class Config:
        """Disabling protected namespaces."""

        protected_namespaces = ()


class Directory(BaseModel):
    """Config for directory."""

    input_root_dir: str = "/data/blob_root/mars-open"
    output_root_dir: str = "/data/amlt_data/mars-exps"


class OrderModel(BaseModel):
    """Config for order model."""

    seq_len: int = 1024
    token_dim: int = 15
    converter_dir: str = "converters-zz1800-500-2024-03-18"


class TradingModelConfig(BaseModel):
    """Config for trading model."""

    model_path: str = "trading-models/model.ckpt"
    updating_model_dir: str = "trading-models/updating_model"
    experience_dir: str = "trading-models/experience"
    log_dir: str = "trading-models/log"
    model_name: str = "trading-model"
    input_dim: int = 26
    output_dim: int = 17
    hidden_dim: int = 128
    num_batches_to_sync_model: int = 100
    target_volume_ratio: float = 0.5
    num_updating_models: int = 5
    lr: float = 1e-3
    num_rollouts: int = 4
    batch_size: int = 256
    buffer_size: int = 1024
    entropy_weight: float = 0.01
    good_completion_ratio: float = 0.95
    base_twap_passive_volume_ratio: float = 0.1


class DebugConfig(BaseModel):
    """Config for debug."""

    enable: bool = False


class Conf(BaseSettings):
    """Config for all."""

    debug: DebugConfig = DebugConfig()
    model_serving: ModelServing = ModelServing()
    order_model: OrderModel = OrderModel()
    trading_model: TradingModelConfig = TradingModelConfig()
    directory: Directory = Directory()

    class Config:
        """Configs."""

        env_nested_delimiter = ENV_NESTED_DELIMITER
        protected_namespaces = ()


# global config
C = Conf()

if __name__ == "__main__":
    # print config
    from rich.pretty import pprint

    pprint(C)
