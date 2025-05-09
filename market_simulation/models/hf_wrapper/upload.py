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
        with file_path.open("rb") as f:
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
    stylized_facts_dir: Path,
    repo_id: str,
    token: str,
    commit_message: str,
) -> None:
    """Upload stylized facts file to Hugging Face Hub.

    Args:
        api: Hugging Face API client
        stylized_facts_dir: Path to the stylized facts directory
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    stylized_facts_path = stylized_facts_dir / "rollout_info_25_minutes.zstd"
    if stylized_facts_path.exists():
        with stylized_facts_path.open("rb") as f:
            stylized_facts_content = f.read()
        api.upload_file(
            path_or_fileobj=stylized_facts_content,
            path_in_repo=f"stylized-facts/{stylized_facts_path.name}",
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
        with readme_path.open("rb") as f:
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


def _upload_validation_files(
    api: HfApi,
    validation_data_path: Path,
    validation_output_path: Path,
    repo_id: str,
    token: str,
    commit_message: str,
) -> None:
    """Upload validation data and output files to Hugging Face Hub.

    Args:
        api: Hugging Face API client
        validation_data_path: Path to the validation data file (.zstd)
        validation_output_path: Path to the validation output file (.csv)
        repo_id: Hugging Face repository ID
        token: Hugging Face API token
        commit_message: Commit message for the upload
    """
    # Upload validation data file
    with validation_data_path.open("rb") as f:
        data_content = f.read()
    api.upload_file(
        path_or_fileobj=data_content,
        path_in_repo=f"validation-samples/{validation_data_path.name}",
        repo_id=repo_id,
        token=token,
        revision="main",
        commit_message=commit_message,
    )
    logging.info(f"Uploaded validation data file {validation_data_path.name} to hub")
    # Upload validation output file
    with validation_output_path.open("rb") as f:
        output_content = f.read()
    api.upload_file(
        path_or_fileobj=output_content,
        path_in_repo=f"validation-samples/{validation_output_path.name}",
        repo_id=repo_id,
        token=token,
        revision="main",
        commit_message=commit_message,
    )
    logging.info(f"Uploaded validation output file {validation_output_path.name} to hub")


def upload_to_hf(
    model_path: Path,
    converter_dir: Path,
    stylized_facts_dir: Path,
    repo_id: str,
    token: str,
    *,
    validation_data_path: Path,
    validation_output_path: Path,
    private: bool = False,
    commit_message: str = "Upload model and converters",
) -> None:
    """Upload model, converters, and optional validation files to Hugging Face Hub.

    Args:
        model_path: Path to the model file.
        converter_dir: Path to the converter directory.
        stylized_facts_dir: Path to the stylized facts directory.
        repo_id: Hugging Face repository ID (e.g., 'microsoft/mars-order-model')
        token: Hugging Face API token
        validation_data_path: Optional path to the validation data file (.zstd).
        validation_output_path: Optional path to the validation output file (.csv).
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    api = HfApi(token=token)

    # Load or create model
    model = _load_or_create_model(model_path)

    # Push the model
    model.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        branch="main",
        commit_message=commit_message,
    )
    logging.info(f"Successfully pushed model to {repo_id}")

    # Upload converters
    _upload_converters(api, converter_dir, repo_id, token, commit_message)

    # Upload stylized facts file
    _upload_stylized_facts(api, stylized_facts_dir, repo_id, token, commit_message)

    # Upload README.md
    _upload_readme(api, repo_id, token, commit_message)

    # Upload validation files
    _upload_validation_files(api, validation_data_path, validation_output_path, repo_id, token, commit_message)

    logging.info(f"Successfully finished upload process to {repo_id}")


def main() -> None:
    """Main function."""
    # Example usage
    model_path_str: str = "2025-04-21/order-time-mid-LOB-24-1024-16.zstd"
    converter_dir_str: str = "converters-zz1800-500-2024-03-18"
    stylized_facts_dir_str: str = "stylized-facts"
    model_path = Path(C.directory.input_root_dir) / model_path_str
    converter_dir = Path(C.directory.input_root_dir) / converter_dir_str
    stylized_facts_dir = Path(C.directory.input_root_dir) / stylized_facts_dir_str
    # Define paths for the validation files
    validation_data_path = Path(C.directory.input_root_dir) / "2025-04-21/valid-00-00000000-0-32.zstd"
    validation_output_path = Path(C.directory.input_root_dir) / "2025-04-21/valid-00-00000000-0-32.output.csv"

    repo_id = input("Please enter your Hugging Face repository ID: ").strip()
    token = input("Please enter your Hugging Face token: ").strip()

    upload_to_hf(
        model_path=model_path,
        converter_dir=converter_dir,
        stylized_facts_dir=stylized_facts_dir,
        validation_data_path=validation_data_path,
        validation_output_path=validation_output_path,
        repo_id=repo_id,
        token=token,
        private=True,  # Set to False for public repository
        commit_message="init commit",
    )


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
