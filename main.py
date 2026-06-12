"""
Training entry script for the unified macro-expression and micro-expression
recognition model.

This script performs the following steps:

    1. Parses command-line arguments.
    2. Sets random seeds for reproducibility.
    3. Selects the training device.
    4. Reads dataset information from a CSV file.
    5. Creates the directory for saving model weights.
    6. Starts Leave-One-Subject-Out training.

Typical usage:

    python main.py \
        --csv_path path/to/data.csv \
        --image_root path/to/images \
        --catego CASME2 \
        --num_classes 3 \
        --batch_size 4 \
        --epochs 100 \
        --learning_rate 1e-3

Notes:
    - The CSV file should contain the dataset metadata.
    - The training function expects a subject column named "Subject".
    - The model uses 51 facial landmarks by default.
"""

import argparse
import ast
import os
import random
import time

import numpy as np
import torch

from read_file import read_csv
from train import loso_train

# ============================================================
# Reproducibility
# ============================================================

SEED_VALUE = 99


def set_seed(seed: int = 99, deterministic: bool = True):
    """
    Set random seeds for reproducible experiments.

    Args:
        seed:
            Random seed value.

        deterministic:
            If True, configures cuDNN to behave deterministically.

    Notes:
        Full reproducibility is not always guaranteed on GPU because some CUDA
        operations can still be nondeterministic depending on the PyTorch and
        CUDA versions.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Argument parsing helpers
# ============================================================


def parse_list(value):
    """
    Parse list-like command-line arguments safely.

    This is useful for arguments such as:

        --model_block_args "[[32, 1, 2], [512, 2, 3]]"
        --kernel_size "[3, 2]"

    Args:
        value:
            A Python list or a string representation of a Python list.

    Returns:
        Parsed Python list.

    Raises:
        argparse.ArgumentTypeError:
            If the input cannot be parsed into a list.
    """

    if isinstance(value, list):
        return value

    try:
        parsed_value = ast.literal_eval(value)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            f"Could not parse list argument: {value}. Error: {error}"
        )

    if not isinstance(parsed_value, list):
        raise argparse.ArgumentTypeError(
            f"Expected a list, but got {type(parsed_value).__name__}: {value}"
        )

    return parsed_value


def parse_bool(value):
    """
    Parse boolean command-line arguments.

    Accepted true values:
        true, 1, yes, y

    Accepted false values:
        false, 0, no, n

    Args:
        value:
            Boolean or string value.

    Returns:
        bool
    """

    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in {"true", "1", "yes", "y"}:
        return True

    if value in {"false", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Expected a boolean value, but got: {value}")


# ============================================================
# Device selection
# ============================================================


def get_device(device_id: int = 0):
    """
    Select training device.

    Args:
        device_id:
            CUDA device index.

    Returns:
        torch.device

    Notes:
        If CUDA is unavailable, CPU is used automatically.
    """

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        if device_id < num_gpus:
            return torch.device(f"cuda:{device_id}")

        print(
            f"Requested cuda:{device_id}, but only {num_gpus} GPU(s) are available. "
            f"Using cuda:0 instead."
        )
        return torch.device("cuda:0")

    return torch.device("cpu")


# ============================================================
# Argument parser
# ============================================================


def build_parser():
    """
    Build and return the command-line argument parser.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Train the unified MaE and ME recognition model."
    )

    # ------------------------------------------------------------
    # Dataset paths
    # ------------------------------------------------------------
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing dataset metadata.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory containing training images or video frames.",
    )

    parser.add_argument(
        "--catego",
        type=str,
        required=True,
        help="Dataset name or category, for example CASME2, CASME3, SAMM, or Mixed.",
    )

    # ------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------
    parser.add_argument(
        "--num_classes",
        type=str,
        default="Folder",
        help=(
            "Number of expression classes. Use 'Folder' if the class number "
            "is inferred from folders."
        ),
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size."
    )

    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs."
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate."
    )

    parser.add_argument(
        "--n_fold",
        type=str,
        default="loso",
        help="Cross-validation setting. Example: loso or 5-fold.",
    )

    parser.add_argument(
        "--patience", type=int, default=50, help="Early stopping patience."
    )

    parser.add_argument(
        "--weight_save_path",
        type=str,
        default="saved models",
        help="Directory for saving model weights.",
    )

    parser.add_argument(
        "--message",
        type=str,
        default="Model",
        help="Short experiment description or model note.",
    )

    parser.add_argument(
        "--device_id", type=int, default=0, help="CUDA device index. Example: 0 or 1."
    )

    # ------------------------------------------------------------
    # Model architecture configuration
    # ------------------------------------------------------------
    parser.add_argument(
        "--act",
        type=str,
        default="nn.ReLU(inplace=True)",
        help="Activation function description. Kept as string for compatibility.",
    )

    parser.add_argument(
        "--stream_input_channel",
        type=int,
        default=49,
        help="Input channel size for the graph stream. Usually 7x7 = 49.",
    )

    parser.add_argument(
        "--model_block_args",
        type=parse_list,
        default=[[32, 1, 2], [512, 2, 3]],
        help=(
            "STFGN block configuration. Each block is [output_channel, stride, depth]."
        ),
    )

    parser.add_argument(
        "--ep_block_args",
        type=parse_list,
        default=[[32, 1, 2], [256, 1, 2], [512, 2, 3]],
        help=(
            "STGAE encoder block configuration. Each block is "
            "[output_channel, stride, depth]."
        ),
    )

    parser.add_argument(
        "--act_type", type=str, default="swish", help="Activation type name."
    )

    parser.add_argument(
        "--layer_type", type=str, default="Sep", help="Temporal layer type."
    )

    parser.add_argument(
        "--drop_prob", type=float, default=0.25, help="Dropout probability."
    )

    parser.add_argument(
        "--kernel_size",
        type=parse_list,
        default=[3, 2],
        help=(
            "Kernel configuration [temporal_window_size, max_graph_distance]. "
            "Example: [3, 2]."
        ),
    )

    parser.add_argument(
        "--reduct_ratio",
        type=int,
        default=4,
        help="Reduction ratio used in Spatio-Temporal Landmark Attention.",
    )

    parser.add_argument(
        "--bias",
        type=parse_bool,
        default=True,
        help="Whether convolution layers use bias.",
    )

    # ------------------------------------------------------------
    # Data representation configuration
    # ------------------------------------------------------------
    parser.add_argument(
        "--num_landmarks",
        type=int,
        default=51,
        help="Number of facial landmarks used in the graph.",
    )

    parser.add_argument(
        "--stream_embedding",
        type=int,
        default=128,
        help="Embedding size for each model stream.",
    )

    parser.add_argument(
        "--num_frames", type=int, default=10, help="Number of selected keyframes."
    )

    parser.add_argument(
        "--num_features",
        type=int,
        default=49,
        help="Feature dimension for each landmark patch. Usually 7x7 = 49.",
    )

    return parser


# ============================================================
# Main training function
# ============================================================


def main():
    """
    Main training pipeline.

    Steps:
        1. Set seed.
        2. Parse arguments.
        3. Select device.
        4. Read CSV data.
        5. Create weight saving directory.
        6. Run LOSO training.
        7. Print total runtime.
    """

    start_time = time.time()

    set_seed(SEED_VALUE, deterministic=True)

    parser = build_parser()
    args = parser.parse_args()

    device = get_device(args.device_id)
    print(f"Using device: {device}")

    # Read dataset metadata and label mapping.
    data, label_mapping = read_csv(args.csv_path, args.num_classes)

    # Create model weight saving directory.
    os.makedirs(args.weight_save_path, exist_ok=True)

    # Start Leave-One-Subject-Out training.
    loso_train(
        data=data,
        sub_column="Subject",
        label_mapping=label_mapping,
        args=args,
        device=device,
        num_classes=args.num_classes,
    )

    elapsed_time = time.time() - start_time
    print(f"Training finished in {elapsed_time / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
