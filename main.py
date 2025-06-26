import argparse
import os
import torch
from read_file import read_csv
from train import loso_train
import time
import random
import numpy as np
start_time = time.time()

SEED_VALUE = 99
np.random.seed(SEED_VALUE)
random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path for the csv file for training data")
    parser.add_argument("--image_root", type=str, required=True, help="Root for the training images")
    parser.add_argument("--catego", type=str, required=True, help="SAMM or CASME^2 dataset")
    parser.add_argument("--num_classes", type=str, default="Folder", help="Classes to be trained")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--weight_save_path", type=str, default="saved models", help="Path for the saving weight")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for training the old model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training the old model")
    parser.add_argument("--n_fold", type=str, default="loso", help="Number of folds for testing, Example:LOSO / 5-foald")
    # parser.add_argument("--n_fold", type=int, default=5, help="Number of folds for testing, Example:LOSO / 5-foald")
    parser.add_argument("--patience", type=int, default=50, help="Number of patience in validation")
    parser.add_argument("--message", type=str, default="Model", help="info about the model")

    parser.add_argument("--act", type=str, default="nn.ReLU(inplace=True)", help="Activation Function")
    parser.add_argument("--stream_input_channel", type=int, default=49, help="info about the model")
    parser.add_argument("--model_block_args", type=list, default=[[32, 1, 2], [512, 2, 3]],
                       help="STGCN model structure [output_channel, stride, depth]")
    parser.add_argument("--ep_block_args", type=list, default=[[32, 1, 2], [256, 1, 2], [512, 2, 3]],
                        help="Graph Autoencoder (Encode) model structure")
    parser.add_argument("--act_type", type=str, default="swish", help="")
    parser.add_argument("--layer_type", type=str, default="Sep", help="Temporal layer type")
    parser.add_argument("--drop_prob", type=int, default=0.25, help="Drop rate")
    parser.add_argument("--kernel_size", type=list, default=[3, 2], help="emporal_window_size, max_graph_distance")
    parser.add_argument("--reduct_ratio", type=int, default=4, help="ST Landmark Attention reduction ratio")
    parser.add_argument("--bias", type=bool, default=True, help="Bias")
    parser.add_argument("--num_landmarks", type=int, default=51, help="Bias")
    parser.add_argument("--stream_embedding", type=int, default=128, help="Bias")
    parser.add_argument("--num_frames", type=int, default=10, help="Bias")
    parser.add_argument("--num_features", type=int, default=49, help="Bias")

    args = parser.parse_args()

    # Training device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Read in the data
    data, label_mapping = read_csv(args.csv_path, args.num_classes)

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    loso_train(data=data, sub_column="Subject", label_mapping=label_mapping, args=args, device=device, num_classes=args.num_classes)


