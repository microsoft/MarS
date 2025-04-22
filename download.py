import logging

from huggingface_hub import snapshot_download

from market_simulation.conf import C
from market_simulation.models.order_model import OrderModel


def cache_and_download() -> None:
    """Download model and converters."""
    # cache model
    model = OrderModel.from_pretrained(C.model_serving.repo_id)
    logging.info(f"Model loaded from {C.model_serving.repo_id}.")
    logging.info(f"Model configs: {model.num_layers}, {model.emb_dim}, {model.num_heads}")

    # download pre-requisites
    snapshot_download(
        repo_id=C.model_serving.repo_id,
        local_dir=C.directory.input_root_dir,
        allow_patterns=["converters/*", "validation-samples/*", "stylized-facts/*"],
    )
    logging.info(f"Pre-requisites downloaded at {C.directory.input_root_dir}.")


if __name__ == "__main__":
    cache_and_download()
