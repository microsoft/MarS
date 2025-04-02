from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import HfApi, PyTorchModelHubMixin

from market_simulation.models.order_model import OrderModel, OrderTokenizer
from market_simulation.utils import pkl_utils


class MarsOrderTokenizer(OrderTokenizer, PyTorchModelHubMixin):
    """Wrapper for OrderTokenizer using PyTorchModelHubMixin for HuggingFace Hub compatibility."""

    def __init__(
        self,
        max_order_index: int,
        emb_dim: int,
        num_max_orders: int,
        num_bins_price_level: int,
        num_bins_pred_order_volume: int,
        num_bins_order_interval: int,
    ) -> None:
        """Initialize the tokenizer with PyTorchModelHubMixin."""
        super().__init__(
            max_order_index=max_order_index,
            emb_dim=emb_dim,
            num_max_orders=num_max_orders,
            num_bins_price_level=num_bins_price_level,
            num_bins_pred_order_volume=num_bins_pred_order_volume,
            num_bins_order_interval=num_bins_order_interval,
        )

    @property
    def config_dict(self) -> dict[str, Any]:
        """Get config dictionary for saving."""
        return {
            "max_order_index": self.max_order_index,
            "emb_dim": self.emb_dim,
            "num_max_orders": self.num_max_orders,
            "num_bins_price_level": self.num_bins_price_level,
            "num_bins_pred_order_volume": self.num_bins_pred_order_volume,
            "num_bins_order_interval": self.num_bins_order_interval,
        }


class MarsOrderModel(OrderModel, PyTorchModelHubMixin):
    """Wrapper for OrderModel using PyTorchModelHubMixin for HuggingFace Hub compatibility."""

    def __init__(
        self,
        emb_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        num_bins_price_level: int = 32,
        num_bins_pred_order_volume: int = 32,
        num_bins_order_interval: int = 16,
        num_max_orders: int = 200,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the model with PyTorchModelHubMixin."""
        super().__init__(
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_bins_price_level=num_bins_price_level,
            num_bins_pred_order_volume=num_bins_pred_order_volume,
            num_bins_order_interval=num_bins_order_interval,
            num_max_orders=num_max_orders,
            dropout=dropout,
        )
        # Initialize tokenizer
        self.order_info_tokenizer = MarsOrderTokenizer(
            max_order_index=self.output_dim,
            emb_dim=emb_dim,
            num_max_orders=num_max_orders,
            num_bins_price_level=num_bins_price_level,
            num_bins_pred_order_volume=num_bins_pred_order_volume,
            num_bins_order_interval=num_bins_order_interval,
        )

    def push_to_hub_func(
        self,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        commit_message: str = "Upload model and tokenizer config",
    ) -> None:
        """Push model and tokenizer config to hub."""
        # First push the model
        super().push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
            branch="main",
            commit_message=commit_message,
        )

        # Then push tokenizer config
        # Create tokenizer config
        tokenizer_config = self.order_info_tokenizer.config_dict
        config_content = json.dumps(tokenizer_config, indent=2)

        # Upload tokenizer config using the API
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=config_content.encode(),
            path_in_repo="tokenizer_config.json",
            repo_id=repo_id,
            token=token,
            revision="main",
            commit_message=commit_message,
        )
        logging.info("Uploaded tokenizer config to hub")

        # Upload README.md
        readme_path = Path(__file__).parent / "README.md"
        if readme_path.exists():
            with open(readme_path, "rb") as f:
                readme_content = f.read()
            api.upload_file(
                path_or_fileobj=readme_content,
                path_in_repo="README.md",
                repo_id=repo_id,
                token=token,
                revision="main",
                commit_message=commit_message,
            )
            logging.info("Uploaded README.md to hub")
        else:
            logging.warning("README.md not found, skipping upload")


def create_random_model() -> MarsOrderModel:
    """Create a randomly initialized model with default parameters."""
    return MarsOrderModel(
        emb_dim=1024,
        num_layers=24,
        num_heads=16,
        num_bins_price_level=32,
        num_bins_pred_order_volume=32,
        num_bins_order_interval=16,
        num_max_orders=200,
        dropout=0.1,
    )


def push_to_hub_mixin(
    model_path: Path | None,
    repo_id: str,
    token: str,
    private: bool = False,
    commit_message: str = "Upload model and tokenizer config",
) -> None:
    """Push model to Hugging Face Hub using PyTorchModelHubMixin.

    Args:
        model_path: Path to the model file. If None, a randomly initialized model will be used.
        repo_id: Hugging Face repository ID (e.g., 'microsoft/mars-order-model')
        token: Hugging Face API token
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    # Load or create model
    if model_path is not None:
        base_model = pkl_utils.load_pkl_zstd(model_path)
        assert isinstance(base_model, OrderModel)

        # Create a new MarsOrderModel with the same parameters
        model = MarsOrderModel(
            emb_dim=base_model.emb_dim,
            num_layers=base_model.num_layers,
            num_heads=base_model.num_heads,
            num_bins_price_level=32,
            num_bins_pred_order_volume=32,
            num_bins_order_interval=16,
            num_max_orders=base_model.num_max_orders,
            dropout=0.1,
        )
        # Copy the state dict from the base model
        model.load_state_dict(base_model.state_dict())
        logging.info(f"Loaded model from {model_path}")
    else:
        model = create_random_model()
        logging.info("Created randomly initialized model")

    # Push to Hub
    model.push_to_hub_func(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
    )
    logging.info(f"Successfully pushed model to {repo_id}")


def main() -> None:
    """Main function."""
    # Example usage
    model_path = None
    repo_id = input("Please enter your Hugging Face repository ID: ").strip()  # Change this to your desired repository ID
    token = input("Please enter your Hugging Face token: ").strip()  # Get token from user input

    push_to_hub_mixin(
        model_path=model_path,
        repo_id=repo_id,
        token=token,
        private=True,  # Set to False for public repository
        commit_message="Mixin model final",
    )


if __name__ == "__main__":
    main()
