from data_utilz.dataloader import (train_val_test_split, get_loader)
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn as nn
from model.MMER_model import MMER_model
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score
import time
import numpy as np
import pickle


def train(epochs: int, patience: int, criterion: nn.Module, optimizer: torch.optim,
          model: nn.Module, scheduler: torch.optim.lr_scheduler,
          train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          model_best_name: str):
    """Train the model

    Parameters
    ----------
    epochs : int
        Epochs for training the model
    model : DSSN
        Model to be trained
    train_loader : DataLoader
        DataLoader to load in the data
    device: torch.device
        Device to be trained on
    model_best_name: str
        Name of the weight file to be saved
    """
    best_accuracy = -1
    best_loss = 100000.0
    wait = 0
    train_losses = []

    for epoch in range(epochs):
        # Set model in training mode
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        # for patches, labels in train_loader:
        for landmark_features, stldn_sequences, labels in train_loader:
            landmark_features = landmark_features.to(device)
            stldn_sequences = stldn_sequences.to(device)
            labels = labels.to(device)

            output = model(landmark_features, stldn_sequences)
            loss = criterion(output, labels)
            train_loss += loss.item()

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            train_accuracy += prediction.sum().item() / labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        train_losses.append(train_loss)

        # validation accuracy
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
                v_loss = criterion(output, labels)
                val_loss += v_loss.item()

                # Compute the accuracy
                prediction = (output.argmax(-1) == labels)
                val_accuracy += prediction.sum().item() / labels.size(0)
                val_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(), average="weighted")

        val_loss = val_loss / len(val_loader)
        val_accuracy = val_accuracy / len(val_loader)
        val_f1_score = val_f1_score / len(val_loader)

        wait += 1

        # print(f" Epoch: {epoch + 1}-> Train [Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} ] | Validation [Loss:{val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1_score:.4f}] ")

        if best_loss > val_loss:
            wait = 0
            torch.save(model.state_dict(), model_best_name)
            best_accuracy = val_accuracy
            best_loss = val_loss
        if wait >= patience:
            break
    return train_losses

def evaluate(test_loader: DataLoader, model: nn.Module,
             device: torch.device):
    """
    Evaluates the performance of a neural network model on a test dataset.

    The function sets the model to evaluation mode and computes various performance metrics including accuracy, 
    Unweighted Average Recall (UAR), Unweighted F1-score (UF1), micro F1-score, and weighted F1-score for the test dataset.

    Parameters:
    test_loader (DataLoader): A DataLoader object providing the test data in batches.
    model (nn.Module): The neural network model to evaluate.
    device (torch.device): The device (CPU or GPU) on which the model and data are loaded.

    Returns:
    tuple: A tuple containing the following metrics averaged over the test dataset:
        - test_accuracy (float): The average accuracy over the test dataset.
        - test_uar (float): The average Unweighted Accuracy (UAR) over the test dataset.
        - test_uf1_score (float): The average Unweighted F1-score (macro F1) over the test dataset.
        - test_microf1_score (float): The average micro F1-score over the test dataset.
        - test_wf1_score (float): The average weighted F1-score over the test dataset.

    """

    # Set into evaluation mode
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

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0) # Accuracy
            test_uar = balanced_accuracy_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy()) #  Unweighted Accuracy (UAR)

            test_uf1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(), average="macro") # Unweighted F1-score
            test_microf1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(), average="micro") # micro F1-score
            test_wf1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(), average="weighted") # weighted F1-score

    return test_accuracy / len(test_loader), test_uar / len(test_loader), test_uf1_score / len(test_loader),  test_microf1_score / len(test_loader), test_wf1_score / len(test_loader)


def loso_train(data: pd.DataFrame, sub_column: str, args, label_mapping: dict, device: torch.device, num_classes):
    """
    Performs Leave-One-Subject-Out (LOSO) cross-validation training on the given dataset.

    The function splits the data into training, validation, and test sets based on the specified number of folds. 
    For each fold, it trains a model using the training set, evaluates it on the validation set, and then tests it 
    on the test set. The performance metrics such as accuracy, Unweighted Average Recall (UAR), Unweighted F1-score (UF1), 
    micro F1-score, and weighted F1-score are calculated for each fold, and the best performing model is saved.

    Parameters:
    data (pd.DataFrame): The dataset to be used for training, validation, and testing.
    sub_column (str): The name of the column in the dataset that identifies the subject for LOSO cross-validation.
    args: A namespace containing various arguments required for training, including batch size, learning rate, and epochs.
    label_mapping (dict): A dictionary that maps labels in the dataset to numerical values.
    device (torch.device): The device (CPU or GPU) on which the model and data will be processed.
    num_classes (int): The number of classes for the classification task.

    Returns:
    None: The function prints and logs the performance metrics of the model for each fold and overall.

    """
    log_file = open("train.log", "w")
    train_list, val_list, test_list = train_val_test_split(data, sub_column, args.n_fold)

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
        # print(f"=================Fold: {idx + 1}=====================")
        start_time = time.time()
        train_csv = train_list[idx]
        val_csv = val_list[idx]
        test_csv = test_list[idx]

        # print(f'In Fold {idx+1} => Total Data: {train_csv.shape[0]+val_csv.shape[0]+test_csv.shape[0]}, Train: {train_csv.shape[0]}, Val: {val_csv.shape[0]}, Test: {test_csv.shape[0]}')

        # Create dataset and dataloader
        _, train_loader = get_loader(csv_file=train_csv,
                                     image_root=args.image_root,
                                     label_mapping=label_mapping,
                                     batch_size=args.batch_size,
                                     device=device,
                                     catego=args.catego,
                                     num_classes=args.num_classes)

        _, val_loader = get_loader(csv_file=val_csv,
                                   image_root=args.image_root,
                                   label_mapping=label_mapping,
                                   # batch_size=len(val_csv),
                                   batch_size=args.batch_size,
                                   device=device,
                                   catego=args.catego,
                                   train=False,
                                   num_classes=args.num_classes)
        _, test_loader = get_loader(csv_file=test_csv,
                                    image_root=args.image_root,
                                    label_mapping=label_mapping,
                                    # batch_size=len(test_csv),
                                    batch_size=args.batch_size,
                                    device=device,
                                    catego=args.catego,
                                    train=False,
                                    shuffle=False,
                                    num_classes=args.num_classes)

        model = MMER_model(args=args, device=device, num_classes=num_classes).to(device)  # aguments

        # Create criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)

        # Train the data
        temp_train_losses = train(epochs=args.epochs,
              patience=args.patience,
              criterion=criterion,
              optimizer=optimizer,
              scheduler=None,
              model=model,
              train_loader=train_loader,
              val_loader=val_loader,
              device=device,
              model_best_name=f"{args.weight_save_path}/model_best.pt")
        model.load_state_dict(torch.load(f"{args.weight_save_path}/model_best.pt", map_location=device))

        temp_test_accuracy, temp_test_uar, temp_test_uf1_score, temp_test_microf1_score, temp_test_wf1_score = evaluate(test_loader=test_loader, model=model, device=device)

        temp_test_accuracy = temp_test_accuracy*100
        temp_test_uar = temp_test_uar * 100


        train_loss_array.append(temp_train_losses)

        if temp_test_accuracy > best_test_accuracy:
            best_test_accuracy = temp_test_accuracy
            train_losses = temp_train_losses

        if test_uar > best_test_uar:
            best_test_uar = test_uar

        print(f"Fold {idx + 1} / Subject : {test_csv['Subject'].unique()})>> ACC: {temp_test_accuracy:.2f}, UAR: {temp_test_uar:.2f}, UF1: {temp_test_uf1_score:.4f}, microF1: {temp_test_microf1_score:.4f}, weightedF1: {temp_test_wf1_score:.4f} ")
        log_file.write(f"Fold {idx + 1} / Subject : {test_csv['Subject'].unique()})>> ACC: {temp_test_accuracy:.2f}, UAR: {temp_test_uar:.2f}, UF1: {temp_test_uf1_score:.4f}, microF1: {temp_test_microf1_score:.4f}, weightedF1: {temp_test_wf1_score:.4f} ")
        # log_file.write(f"Fold {idx + 1}>> ACC: {temp_test_accuracy:.2f}, UAR: {temp_test_uar:.2f}, UF1: {temp_test_uf1_score:.4f}, microF1: {temp_test_microf1_score:.4f}, weightedF1: {temp_test_wf1_score:.4f} \n")
        test_accuracy += temp_test_accuracy
        test_uar += temp_test_uar
        test_uf1_score += temp_test_uf1_score
        test_microf1_score += temp_test_microf1_score
        test_wf1_score += temp_test_wf1_score
        end_time = time.time() - start_time
        # print(f"--- Fold {idx + 1} time ---{int(end_time / 60)}min {end_time % 60:.0f}sec")
    print(
        f"Total>> {args.catego}_{args.num_classes}_{args.message} - ACC: {test_accuracy / len(train_list):.2f}, UAR: {test_uar / len(train_list):.2f}, UF1: {test_uf1_score / len(train_list):.4f}, microF1: {test_microf1_score / len(train_list):.4f}, weightedF1: {test_wf1_score / len(train_list):.4f}")

    log_file.write(f"Total>> {args.catego}_{args.num_classes}_{args.message} - ACC: {test_accuracy / len(train_list):.2f}, UAR: {test_uar / len(train_list):.2f}, UF1: {test_uf1_score / len(train_list):.4f}, microF1: {test_microf1_score / len(train_list):.4f}, weightedF1: {test_wf1_score / len(train_list):.4f}")



