
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


# 重构后的版本 - 直接使用预先生成的路径

def process_audio_data_to_pickle(dataset_name: str, in_filename: str, out_filename: str, extractor):
    """
    从 DataFrame 中提取音频特征，该 DataFrame 已包含完整的音频文件路径。
    """
    
    preprocessed_dir = CONFIG.dataset_preprocessed_dir_path(dataset_name)
    dataframe = pd.read_pickle(os.path.join(preprocessed_dir, in_filename))

    # 辅助函数现在变得极其简单
    def _extract_data_from_audio(audio_path: str):
                # ▼▼▼ 在每个 worker 的第一次任务中加载配置 ▼▼▼
        global _worker_config_loaded
        if not _worker_config_loaded:
            CONFIG.load_config("config.yaml") # 确保配置文件路径正确
            _worker_config_loaded = True
        try:
            # 路径有效性检查
            if not os.path.exists(audio_path):
                print(f"[WARNING] Audio file not found: {audio_path}. Skipping.")
                return None
            return extractor.extract(audio_path)
        except Exception as e:
            print(f"[ERROR] Failed to extract features for {audio_path}: {e}")
            return None

    # 使用 joblib 并行处理，直接迭代 audio_path 列
    # 这比之前的 for 循环快得多
    extracted_features = Parallel(n_jobs=-1)(
        delayed(_extract_data_from_audio)(path)
        for path in tqdm(dataframe["audio_path"], desc=f"Extracting features for {dataset_name}")
    )

    # 将提取的特征添加到 DataFrame，并移除处理失败的行
    dataframe["audio_features"] = extracted_features
    dataframe.dropna(subset=['audio_features'], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    # 如果没有数据，则提前退出
    if len(dataframe) == 0:
        print(f"[WARNING] No audio features were extracted for {dataset_name}.")
        dataframe.to_pickle(os.path.join(preprocessed_dir, out_filename))
        return

    # 对特征进行填充 (Padding)
    max_length = dataframe["audio_features"].apply(lambda x: x.shape[0]).max()
    dataframe["audio"] = dataframe["audio_features"].apply(
        lambda x: np.pad(x, (0, max_length - x.shape[0]), "constant")
    )
    
    # 清理 DataFrame 并保存
    dataframe.drop(columns=['audio_features', 'audio_path', 'audio_filename'], inplace=True, errors='ignore')
    dataframe.to_pickle(os.path.join(preprocessed_dir, out_filename))
    print(f"[INFO] Audio feature DataFrame for {dataset_name} saved. Shape: {dataframe.shape}")

def process_text_data_to_pickle(dataset_name: str, in_filename: str, out_filename: str, tokenizer):
    """
    Tokenizes text data for a given dataset and saves the resulting DataFrame to a pickle file.

    Args:
        dataset_name (str): The name of the dataset being processed.
        in_filename (str): The name of the input pickle file containing data with text.
        out_filename (str): The name of the output pickle file.
        tokenizer: A text tokenizer instance.
    """
    preprocessed_dir = CONFIG.dataset_preprocessed_dir_path(dataset_name)
    dataframe = pd.read_pickle(os.path.join(preprocessed_dir, in_filename))

    if len(dataframe) == 0:
        print(f"[WARNING] No data to process text for dataset: {dataset_name}")
        # Save an empty DataFrame to the output file
        empty_df = pd.DataFrame(columns=dataframe.columns)
        output_filepath = os.path.join(preprocessed_dir, out_filename)
        empty_df.to_pickle(output_filepath)
        print(f"[INFO] Empty text DataFrame saved to: {output_filepath}")
        return


    # Determine max text length based on the current dataset's text column
    max_text_length = dataframe["text"].apply(len).max()
    print(f"[INFO] Max text length for {dataset_name}: {max_text_length}")


    dataframe["text"] = dataframe["text"].apply(
        lambda text: np.array(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=max_text_length,
            )
        )
    )
    output_filepath = os.path.join(preprocessed_dir, out_filename)
    dataframe.to_pickle(output_filepath)
    print(f"[INFO] Text tokenized DataFrame saved to: {output_filepath}")
