
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
import torchaudio

from transformers import Wav2Vec2Processor, WavLMModel, Wav2Vec2FeatureExtractor, AutoTokenizer


from joblib import Parallel, delayed
from tqdm import tqdm

from core.config import CONFIG
from preprocessing.iemocap import IemocapPreprocessor
from preprocessing.cremad import CremaDPreprocessor # Assuming CremaDPreprocessor is in preprocessing.cremad or similar


_worker_config_loaded = False


def process_raw_data_to_pickle(dataset_name: str, out_filename: str):
    """
    Processes raw data for a given dataset and saves the resulting DataFrame to a pickle file.

    Args:
        dataset_name (str): The name of the dataset to process (e.g., "IEMOCAP_full_release1-3", "CREMA-D").
        out_filename (str): The name of the output pickle file.
    """
    dataset_path = CONFIG.dataset_path(dataset_type="training" if dataset_name == CONFIG.training_dataset_name() else "evaluation") # Determine path based on dataset type
    preprocessed_path = CONFIG.dataset_preprocessed_dir_path(dataset_name)

    if dataset_name == CONFIG.training_dataset_name():
        print(f"[INFO] Using IemocapPreprocessor for dataset: {dataset_name}")
        preprocessor = IemocapPreprocessor(dataset_path)
    elif dataset_name == CONFIG.evaluation_dataset_name():
        print(f"[INFO] Using CremaDPreprocessor for dataset: {dataset_name}")
        # Assuming CremaDPreprocessor class is accessible in this scope or imported
        # If not already defined or imported, make sure it is available.
        try:
            preprocessor = CremaDPreprocessor(dataset_path)
        except NameError:
            print("[ERROR] CremaDPreprocessor not found. Please ensure it is defined or imported.")
            return # Exit the function if CremaDPreprocessor is not available
    else:
        print(f"[ERROR] Unknown dataset name specified: {dataset_name}")
        return # Exit the function for unknown datasets

    df = preprocessor.generate_dataframe()

    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    output_filepath = os.path.join(preprocessed_path, out_filename)
    df.to_pickle(output_filepath)
    print(f"[INFO] Raw data DataFrame saved to: {output_filepath}")


