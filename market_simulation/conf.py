from pydantic import BaseModel
from pydantic_settings import BaseSettings

ENV_NESTED_DELIMITER = "__"


class ModelServing(BaseModel):
    """Config for model serving."""

    repo_id: str = "microsoft/mars-order-model"
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
    converter_dir: str = "converters"


class DebugConfig(BaseModel):
    """Config for debug."""

    enable: bool = True


class Conf(BaseSettings):
    """Config for all."""

    debug: DebugConfig = DebugConfig()
    model_serving: ModelServing = ModelServing()
    order_model: OrderModel = OrderModel()
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
