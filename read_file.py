import numpy as np
import pandas as pd


def read_csv(path: str, num_classes) -> tuple:
    """Read in the csv from given path,
    and return the label mapping

    Parameters
    ----------
    path : str
        Path for CSV file

    Returns
    -------
    Tuple
        Return data and label mapping
    """
    # Read in dataset
    data = pd.read_csv(path, dtype={"Subject": str})
    if num_classes == "Folder":
        data.loc[data['Filename'].str.contains('anger'), 'Estimated Emotion Folder'] = 'anger'
        data.loc[data['Filename'].str.contains('disgust'), 'Estimated Emotion Folder'] = 'disgust'
        data.loc[data['Filename'].str.contains('happy'), 'Estimated Emotion Folder'] = 'happy'

    emotion_categories = "Estimated Emotion " + str(num_classes)
    data.dropna(subset=[emotion_categories], inplace=True)
    data.reset_index(inplace=True)


    # Label the emotions into number
    label_mapping = {
        emotion: idx for idx, emotion in enumerate(np.unique(data.loc[:, emotion_categories]))
    }
    print(label_mapping)

    return data, label_mapping