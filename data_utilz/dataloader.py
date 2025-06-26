# from dataset import MMDataset
from data_utilz.dataset import MMDataset

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader



def train_val_test_split(data: pd.DataFrame, sub_column: str, n_fold="loso") -> tuple:
    ##To arrange the training and testing set, we have randomly partitioned the available benchmarked datasets in offline
    # mode with a ratio of 80:20 respectively. Furthermore, training set is also divided into training and validation set
    # with 70:30 ratio respectively. Since, the data partition is done randomly, therefore, to make fair analysis of the
    # outcomes, we repeat the same experiment 5 times and then average recognition rate is considered as the final accuracy.
    # Recognition accuracy is calculated by the following:
    # Recog.Accur.=(Total no. o f correctly predicted samples/ Total no. o f samples) × 100

    """Generate train, validation and test data using leave-one-subject-out for
    specific subject

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame
    sub_column : str
        Subject column in DataFrame

    Returns
    -------
    tuple
        Return training, validation and testing list DataFrame
    """

    # Save the training and testing list for all subject
    train_list = []
    val_list = []
    test_list = []

    # Unique subject in `sub_column`
    subjects = np.unique(data[sub_column])

    if n_fold=='loso':

        len_subject = (subjects.shape[0])
        val_count = int(np.ceil(len_subject * 0.2))

        for index, test_subject in enumerate(subjects):
            # Mask for the training
            val_subject = []
            for i in range(val_count):
                val_index = (index + i + 1) % len_subject
                val_subject.append(subjects[val_index])
            # val_subject = subjects[((index+1)%len_subject)]

            mask = data["Subject"].isin(val_subject)
            val_data = data[mask].reset_index(drop=True)  # val_data include val_subject

            mask = data["Subject"].isin([test_subject])
            test_data = data[mask].reset_index(drop=True)  # test_data include test_subject

            val_subject.append(test_subject)

            mask = data["Subject"].isin(val_subject)
            train_data = data[~mask].reset_index(drop=True) # train_data exclude test_subject and val_subject


            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)
    else:

        n_fold = int(n_fold)
        print(f'>>>>>>>>>>>{n_fold}')
        train_test_ratio = 1.-(1./n_fold)
        train_val_ration = 0.7
        train_num = int(np.ceil(subjects.shape[0] * train_test_ratio))
        val_num = int(np.ceil(train_num * train_val_ration))


        for flod in range(n_fold):
        # partition the dataset into (train:70, val:30):80 and test:20
            np.random.seed(flod)
            np.random.shuffle(subjects)

            train_index = subjects[:train_num]
            val_index = subjects[val_num:train_num]
            test_index = subjects[val_num:]
            # print(f'{flod=} {train_index}')

            mask = data["Subject"].isin(train_index)
            train_data = data[mask].reset_index(drop=True)
            mask = data["Subject"].isin(val_index)
            val_data = data[mask].reset_index(drop=True)
            mask = data["Subject"].isin(test_index)
            test_data = data[mask].reset_index(drop=True)

            train_list.append(train_data)
            val_list.append(val_data)
            test_list.append(test_data)

    return train_list, val_list, test_list


def get_loader(csv_file, label_mapping,
               image_root, batch_size,
               catego, device, train=True,
               shuffle=True, num_classes=5):
    dataset = MMDataset(data_info=csv_file,
                        label_mapping=label_mapping,
                        image_root=image_root,
                        catego=catego,
                        device=device,
                        train=train, num_classes=num_classes)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True)

    return dataset, dataloader


