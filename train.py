"""
Training and Leave-One-Subject-Out evaluation utilities.

This file contains the main training, validation, testing, and LOSO
cross-validation functions for the unified macro-expression and
micro-expression recognition model.

Main functions:
    train()
        Trains the model on one training split and saves the best checkpoint
        based on validation loss.

    evaluate()
        Evaluates the trained model on one test split.

    loso_train()
        Runs Leave-One-Subject-Out cross-validation.

Expected data from each DataLoader:
    Each batch should return:

        landmark_features, stldn_sequences, labels

    landmark_features:
        Tensor used by the Facial Graph Stream.

        Expected shape:
            [B, T * V, S, S]

        where:
            B = batch size
            T = number of STLDN maps
            V = number of facial landmarks, usually 51
            S = landmark patch size, usually 7

    stldn_sequences:
        Tensor used by the Visual Stream.

        Expected shape:
            [B, T, H, W]

        where:
            H = STLDN image height
            W = STLDN image width

    labels:
        Ground-truth class labels.

        Expected shape:
            [B]

Metrics:
    ACC:
        Accuracy.

    UAR:
        Unweighted Average Recall.
        Implemented using balanced_accuracy_score.

    UF1:
        Unweighted F1 score.
        Implemented using macro F1 score.

    microF1:
        Micro-averaged F1 score.

    weightedF1:
        Weighted F1 score.
"""

import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)
from torch.utils.data import DataLoader

from data_utilz.dataloader import (
    get_loader,
    train_val_test_split,
)
from model.MMER_model import MMER_model


def train(
    epochs: int,
    patience: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_best_name: str,
):
    """
    Train the model for one train-validation split.

    This function trains the model using the training loader and evaluates it
    on the validation loader after each epoch. The model checkpoint with the
    lowest validation loss is saved.

    Early stopping is controlled by the patience parameter. If the validation
    loss does not improve for `patience` consecutive epochs, training stops.

    Args:
        epochs:
            Maximum number of training epochs.

        patience:
            Number of epochs to wait without validation loss improvement before
            stopping training.

        criterion:
            Loss function.

            Example:
                nn.CrossEntropyLoss()

        optimizer:
            Optimizer used for model parameter updates.

            Example:
                torch.optim.Adam(model.parameters(), lr=1e-4)

        model:
            Model to train.

            In this project:
                MMER_model

        scheduler:
            Learning-rate scheduler.

            If None, no scheduler step is applied.

        train_loader:
            DataLoader for training samples.

            Each batch should return:
                landmark_features, stldn_sequences, labels

        val_loader:
            DataLoader for validation samples.

        device:
            Device used for training.

            Example:
                torch.device("cuda:0")
                torch.device("cpu")

        model_best_name:
            File path where the best model checkpoint is saved.

    Returns:
        train_losses:
            List of average training loss values, one value per epoch.

    Notes:
        The best model is selected based on validation loss, not validation
        accuracy.

        If your model classifier already includes Softmax, then
        nn.CrossEntropyLoss is not ideal because CrossEntropyLoss expects raw
        logits. In that case, remove Softmax from the model classifier.
    """

    best_accuracy = -1
    best_loss = 100000.0
    wait = 0
    train_losses = []

    for epoch in range(epochs):
        # ------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------
        model.train()

        train_loss = 0.0
        train_accuracy = 0.0

        for landmark_features, stldn_sequences, labels in train_loader:
            # Move batch data to the selected device.
            landmark_features = landmark_features.to(device)
            stldn_sequences = stldn_sequences.to(device)
            labels = labels.to(device)

            # Forward pass.
            output = model(landmark_features, stldn_sequences)

            # Compute loss.
            loss = criterion(output, labels)
            train_loss += loss.item()

            # Backpropagation.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute batch accuracy.
            prediction = output.argmax(dim=-1)
            correct = prediction.eq(labels).sum().item()
            train_accuracy += correct / labels.size(0)

        # Step the learning-rate scheduler, if available.
        if scheduler is not None:
            scheduler.step()

        # Average training metrics over all batches.
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        train_losses.append(train_loss)

        # ------------------------------------------------------------
        # Validation phase
        # ------------------------------------------------------------
        model.eval()

        val_accuracy = 0.0
        val_f1_score = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for landmark_features, stldn_sequences, labels in val_loader:
                landmark_features = landmark_features.to(device)
                stldn_sequences = stldn_sequences.to(device)
                labels = labels.to(device)

                output = model(landmark_features, stldn_sequences)

                # Validation loss.
                v_loss = criterion(output, labels)
                val_loss += v_loss.item()

                # Validation accuracy.
                prediction = output.argmax(dim=-1)
                correct = prediction.eq(labels).sum().item()
                val_accuracy += correct / labels.size(0)

                # Weighted F1 score.
                val_f1_score += f1_score(
                    labels.cpu().numpy(),
                    prediction.cpu().numpy(),
                    average="weighted",
                )

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_f1_score /= len(val_loader)

        wait += 1

        # Optional logging.
        # print(
        #     f"Epoch {epoch + 1:03d} | "
        #     f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
        #     f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
        #     f"Val F1: {val_f1_score:.4f}"
        # )

        # Save checkpoint if validation loss improves.
        if best_loss > val_loss:
            wait = 0
            best_accuracy = val_accuracy
            best_loss = val_loss

            torch.save(
                model.state_dict(),
                model_best_name,
            )

        # Early stopping.
        if wait >= patience:
            break

    return train_losses


def evaluate(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
):
    """
    Evaluate the model on a test set.

    This function computes classification performance on the test split.

    Args:
        test_loader:
            DataLoader for test samples.

            Each batch should return:
                landmark_features, stldn_sequences, labels

        model:
            Trained model.

        device:
            Device used for evaluation.

    Returns:
        test_accuracy:
            Average accuracy over the test DataLoader.

        test_uar:
            Unweighted Average Recall.

        test_uf1_score:
            Unweighted F1 score, also called macro F1.

        test_microf1_score:
            Micro F1 score.

        test_wf1_score:
            Weighted F1 score.

    Important:
        The current implementation computes metrics batch by batch and then
        averages them. This is acceptable when all batches have the same size,
        but for exact dataset-level metrics, collect all predictions and labels
        first, then compute metrics once.
    """

    model.eval()

    test_accuracy = 0.0
    test_uar = 0.0
    test_uf1_score = 0.0
    test_microf1_score = 0.0
    test_wf1_score = 0.0

    with torch.no_grad():
        for landmark_features, stldn_sequences, labels in test_loader:
            landmark_features = landmark_features.to(device)
            stldn_sequences = stldn_sequences.to(device)
            labels = labels.to(device)

            output = model(landmark_features, stldn_sequences)

            prediction = output.argmax(dim=-1)

            # Accuracy.
            correct = prediction.eq(labels).sum().item()
            test_accuracy += correct / labels.size(0)

            # Convert to NumPy arrays for sklearn metrics.
            y_true = labels.cpu().numpy()
            y_pred = prediction.cpu().numpy()

            # UAR.
            test_uar += balanced_accuracy_score(y_true, y_pred)

            # UF1, micro F1, and weighted F1.
            test_uf1_score += f1_score(y_true, y_pred, average="macro")
            test_microf1_score += f1_score(y_true, y_pred, average="micro")
            test_wf1_score += f1_score(y_true, y_pred, average="weighted")

    num_batches = len(test_loader)

    return (
        test_accuracy / num_batches,
        test_uar / num_batches,
        test_uf1_score / num_batches,
        test_microf1_score / num_batches,
        test_wf1_score / num_batches,
    )


def loso_train(
    data: pd.DataFrame,
    sub_column: str,
    args,
    label_mapping: dict,
    device: torch.device,
    num_classes,
):
    """
    Run Leave-One-Subject-Out cross-validation.

    In LOSO evaluation, one subject is used as the test subject in each fold.
    The remaining subjects are used for training and validation.

    Processing steps for each fold:
        1. Split the dataset into train, validation, and test sets.
        2. Build DataLoaders.
        3. Initialize a new MMER_model.
        4. Train the model.
        5. Load the best validation-loss checkpoint.
        6. Evaluate on the test subject.
        7. Accumulate metrics across folds.

    Args:
        data:
            Full dataset metadata as a pandas DataFrame.

        sub_column:
            Column name that contains subject IDs.

            Example:
                "Subject"

        args:
            Configuration namespace.

            Expected attributes:
                args.n_fold
                args.image_root
                args.batch_size
                args.catego
                args.num_classes
                args.learning_rate
                args.epochs
                args.patience
                args.weight_save_path
                args.message

        label_mapping:
            Dictionary mapping emotion labels to integer class IDs.

            Example:
                {
                    "happy": 0,
                    "angry": 1,
                    "disgust": 2
                }

        device:
            Training device.

        num_classes:
            Number of expression classes or "Folder".

    Returns:
        None.

        The function prints fold-wise metrics and final average metrics.
        It also writes the same information to train.log.

    Output log:
        train.log

    Checkpoint:
        The best model for each fold is saved to:

            {args.weight_save_path}/model_best.pt

        Note:
            This file is overwritten in every fold.
            If you want to keep all fold checkpoints, include the fold index
            in the checkpoint filename.
    """

    log_file = open("train.log", "w")

    # Generate LOSO or fold-based train, validation, and test splits.
    train_list, val_list, test_list = train_val_test_split(
        data,
        sub_column,
        args.n_fold,
    )

    # Accumulators for average metrics across folds.
    test_accuracy = 0.0
    test_uar = 0.0
    test_uf1_score = 0.0
    test_microf1_score = 0.0
    test_wf1_score = 0.0

    best_test_uar = 0.0
    best_test_accuracy = 0.0
    train_losses = 0

    train_loss_array = []

    for idx in range(len(train_list)):
        start_time = time.time()

        train_csv = train_list[idx]
        val_csv = val_list[idx]
        test_csv = test_list[idx]

        # ------------------------------------------------------------
        # Create train, validation, and test DataLoaders
        # ------------------------------------------------------------
        _, train_loader = get_loader(
            csv_file=train_csv,
            image_root=args.image_root,
            label_mapping=label_mapping,
            batch_size=args.batch_size,
            device=device,
            catego=args.catego,
            num_classes=args.num_classes,
        )

        _, val_loader = get_loader(
            csv_file=val_csv,
            image_root=args.image_root,
            label_mapping=label_mapping,
            batch_size=args.batch_size,
            device=device,
            catego=args.catego,
            train=False,
            num_classes=args.num_classes,
        )

        _, test_loader = get_loader(
            csv_file=test_csv,
            image_root=args.image_root,
            label_mapping=label_mapping,
            batch_size=args.batch_size,
            device=device,
            catego=args.catego,
            train=False,
            shuffle=False,
            num_classes=args.num_classes,
        )

        # ------------------------------------------------------------
        # Initialize model for the current fold
        # ------------------------------------------------------------
        model = MMER_model(
            args=args,
            device=device,
            num_classes=num_classes,
        ).to(device)

        # Loss function.
        criterion = nn.CrossEntropyLoss()

        # Optimizer.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )

        # ------------------------------------------------------------
        # Train the model
        # ------------------------------------------------------------
        checkpoint_path = f"{args.weight_save_path}/model_best.pt"

        temp_train_losses = train(
            epochs=args.epochs,
            patience=args.patience,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_best_name=checkpoint_path,
        )

        # Load the best checkpoint selected by validation loss.
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )

        # ------------------------------------------------------------
        # Evaluate on the test subject
        # ------------------------------------------------------------
        (
            temp_test_accuracy,
            temp_test_uar,
            temp_test_uf1_score,
            temp_test_microf1_score,
            temp_test_wf1_score,
        ) = evaluate(
            test_loader=test_loader,
            model=model,
            device=device,
        )

        # Convert accuracy and UAR to percentage.
        temp_test_accuracy *= 100
        temp_test_uar *= 100

        train_loss_array.append(temp_train_losses)

        # Track the fold with the best test accuracy.
        if temp_test_accuracy > best_test_accuracy:
            best_test_accuracy = temp_test_accuracy
            train_losses = temp_train_losses

        # Track the best test UAR.
        if temp_test_uar > best_test_uar:
            best_test_uar = temp_test_uar

        # ------------------------------------------------------------
        # Print and log fold result
        # ------------------------------------------------------------
        fold_time = time.time() - start_time

        fold_message = (
            f"Fold {idx + 1} / Subject: {test_csv['Subject'].unique()} >> "
            f"ACC: {temp_test_accuracy:.2f}, "
            f"UAR: {temp_test_uar:.2f}, "
            f"UF1: {temp_test_uf1_score:.4f}, "
            f"microF1: {temp_test_microf1_score:.4f}, "
            f"weightedF1: {temp_test_wf1_score:.4f}, "
            f"Time: {int(fold_time / 60)}min {fold_time % 60:.0f}sec"
        )

        print(fold_message)
        log_file.write(fold_message + "\n")

        # Accumulate metrics.
        test_accuracy += temp_test_accuracy
        test_uar += temp_test_uar
        test_uf1_score += temp_test_uf1_score
        test_microf1_score += temp_test_microf1_score
        test_wf1_score += temp_test_wf1_score

    # ------------------------------------------------------------
    # Print and log final average result
    # ------------------------------------------------------------
    num_folds = len(train_list)

    total_message = (
        f"Total >> {args.catego}_{args.num_classes}_{args.message} | "
        f"ACC: {test_accuracy / num_folds:.2f}, "
        f"UAR: {test_uar / num_folds:.2f}, "
        f"UF1: {test_uf1_score / num_folds:.4f}, "
        f"microF1: {test_microf1_score / num_folds:.4f}, "
        f"weightedF1: {test_wf1_score / num_folds:.4f}"
    )

    print(total_message)
    log_file.write(total_message + "\n")
    log_file.close()
