import logging
from pathlib import Path

from huggingface_hub import HfApi

from market_simulation.conf import C
from market_simulation.models.order_model import OrderModel
from market_simulation.utils import pkl_utils


def _load_or_create_model(model_path: Path | None) -> OrderModel:
    """Load an existing model or create a new one with default parameters.

    Args:
        model_path: Path to the model file. If None, a randomly initialized model will be used.

    Returns:
        The loaded or newly created OrderModel.
    """
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

    return model


def _upload_converters(
    api: HfApi,
    converter_dir: Path | None,
    repo_id: str,
    token: str,
    commit_message: str,
) -> None:
    """Upload converter files to Hugging Face Hub.

    Args:
        api: Hugging Face API client
        converter_dir: Path to the converter directory
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    if converter_dir is None or not converter_dir.exists():
        logging.warning("Converter directory not provided or does not exist, skipping converter upload")
        return

    # List of required converter files
    converter_files = [
        "price.zstd",
        "price-level.zstd",
        "price-change-ratio.zstd",
        "order-volume.zstd",
        "lob-volume.zstd",
        "pred-order-volume.zstd",
        "order-interval.zstd",
        "minute-buy-order-count.zstd",
        "minute-trans-vwap-change.zstd",
        "minute-trans-volume.zstd",
        "num-minutes-to-open.zstd",
        "minute-cancel-volume.zstd",
        "lob-spread.zstd",
    ]

    # Verify all required files exist
    missing_files = []
    for file in converter_files:
        if not (converter_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        raise ValueError(f"Missing required converter files: {', '.join(missing_files)}")

    # Upload each converter file directly from source
    for file in converter_files:
        file_path = converter_dir / file
        with Path.open(file_path, "rb") as f:
            file_content = f.read()

        api.upload_file(
            path_or_fileobj=file_content,
            path_in_repo=f"converters/{file}",
            repo_id=repo_id,
            token=token,
            revision="main",
            commit_message=commit_message,
        )
        logging.info(f"Uploaded {file} to hub")

    logging.info("Successfully uploaded converters to hub")


def _upload_stylized_facts(
    api: HfApi,
    repo_id: str,
    token: str,
    commit_message: str,
) -> None:
    """Upload stylized facts file to Hugging Face Hub.

    Args:
        api: Hugging Face API client
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    stylized_facts_path = Path(C.directory.input_root_dir) / "stylized-facts/rollout_info_25_minutes.zstd"
    if stylized_facts_path.exists():
        with Path.open(stylized_facts_path, "rb") as f:
            stylized_facts_content = f.read()
        api.upload_file(
            path_or_fileobj=stylized_facts_content,
            path_in_repo="stylized-facts/rollout_info_25_minutes.zstd",
            repo_id=repo_id,
            token=token,
            revision="main",
            commit_message=commit_message,
        )
        logging.info("Uploaded stylized facts file to hub")
    else:
        logging.warning("Stylized facts file not found, skipping upload")


def _upload_readme(
    api: HfApi,
    repo_id: str,
    token: str,
    commit_message: str,
) -> None:
    """Upload README.md to Hugging Face Hub.

    Args:
        api: Hugging Face API client
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with Path.open(readme_path, "rb") as f:
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


def upload_to_hf(
    model_path: Path | None,
    converter_dir: Path | None,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    commit_message: str = "Upload model and converters",
) -> None:
    """Upload model and converters to Hugging Face Hub.

    Args:
        model_path: Path to the model file. If None, a randomly initialized model will be used.
        converter_dir: Path to the converter directory. If None, converters will not be uploaded.
        repo_id: Hugging Face repository ID (e.g., 'microsoft/mars-order-model')
        token: Hugging Face API token
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    api = HfApi(token=token)

    # Load or create model
    model = _load_or_create_model(model_path)

    # Convert model to FP16
    model.half()
    logging.info("Converted model to FP16 precision.")

    # Push the model
    model.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        branch="main",
        commit_message=commit_message,
    )

    # Upload converters
    _upload_converters(api, converter_dir, repo_id, token, commit_message)

    # Upload stylized facts file
    _upload_stylized_facts(api, repo_id, token, commit_message)

    # Upload README.md
    _upload_readme(api, repo_id, token, commit_message)

    logging.info(f"Successfully pushed model and converters to {repo_id}")


def main() -> None:
    """Main function."""
    # Example usage
    model_path = Path(C.directory.input_root_dir) / C.model_serving.model_path
    converter_dir = Path(C.directory.input_root_dir) / C.order_model.converter_dir
    repo_id = input("Please enter your Hugging Face repository ID: ").strip()  # Change this to your desired repository ID
    token = input("Please enter your Hugging Face token: ").strip()  # Get token from user input

    upload_to_hf(
        model_path=model_path,
        converter_dir=converter_dir,
        repo_id=repo_id,
        token=token,
        private=True,  # Set to False for public repository
        commit_message="init commit",
    )


if __name__ == "__main__":
    main()
