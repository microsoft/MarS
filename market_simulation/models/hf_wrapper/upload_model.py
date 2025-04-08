import logging
from pathlib import Path

from huggingface_hub import HfApi

from market_simulation.models.order_model import OrderModel
from market_simulation.utils import pkl_utils


def upload_to_hf(
    model_path: Path | None,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    commit_message: str = "Upload model",
) -> None:
    """Upload model to Hugging Face Hub.

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

        # Create a new OrderModel with the same parameters
        model = OrderModel(
            emb_dim=base_model.emb_dim,
            num_layers=base_model.num_layers,
            num_heads=base_model.num_heads,
            num_bins_price_level=base_model.order_info_tokenizer.num_bins_price_level,
            num_bins_pred_order_volume=base_model.order_info_tokenizer.num_bins_pred_order_volume,
            num_bins_order_interval=base_model.order_info_tokenizer.num_bins_order_interval,
            num_max_orders=base_model.num_max_orders,
            dropout=0.1,
        )
        # Copy the state dict from the base model
        model.load_state_dict(base_model.state_dict())
        logging.info(f"Loaded model from {model_path}")
    else:
        # Create a randomly initialized model with default parameters
        model = OrderModel(
            emb_dim=1024,
            num_layers=24,
            num_heads=16,
            num_bins_price_level=32,
            num_bins_pred_order_volume=32,
            num_bins_order_interval=16,
            num_max_orders=1024,
            dropout=0.1,
        )
        logging.info("Created randomly initialized model")

    # Push the model
    model.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        branch="main",
        commit_message=commit_message,
    )

    # Upload README.md if it exists
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with Path.open(readme_path, "rb") as f:
            readme_content = f.read()
        api = HfApi(token=token)
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

    logging.info(f"Successfully pushed model to {repo_id}")


def main() -> None:
    """Main function."""
    # Example usage
    model_path = Path("/data/blob_root/mars-open/order-model/order-time-mid-LOB-24-1024-16.zstd")
    repo_id = input("Please enter your Hugging Face repository ID: ").strip()  # Change this to your desired repository ID
    token = input("Please enter your Hugging Face token: ").strip()  # Get token from user input

    upload_to_hf(
        model_path=model_path,
        repo_id=repo_id,
        token=token,
        private=True,  # Set to False for public repository
        commit_message="Initial Mixin Model",
    )


if __name__ == "__main__":
    main()
