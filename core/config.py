import os
import torch
import yaml
from typing import List

# This part doesn't need to change
device = "cuda" if torch.cuda.is_available() else "cpu"

class CONFIG:
    _dict: dict

    @classmethod
    def load_config(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as file:
            cls._dict = yaml.safe_load(file)

    # New methods to get training and evaluation dataset names
    @classmethod
    def training_dataset_name(cls):
        return cls._dict["dataset"]["training_set"]

    @classmethod
    def evaluation_dataset_name(cls):
        return cls._dict["dataset"]["evaluation_set"]

    @classmethod
    def project_root(cls):
        return cls._dict["project_root"]

    @classmethod
    def _path_from_data(cls, path: str):
        base_data_path = cls._dict["data_source"]["path"]
        # if os.path.isabs(base_data_path):
        #     return os.path.join(base_data_path, path)
        # else:
        return os.path.join(cls.project_root(), base_data_path, path)

    # Modified dataset_path to return path based on current dataset context (training/evaluation)
    # This might need further refinement depending on how it's used later
    @classmethod
    def dataset_path(cls, dataset_type: str = "training"):
        if dataset_type == "training":
            dataset_name = cls.training_dataset_name()
        elif dataset_type == "evaluation":
            dataset_name = cls.evaluation_dataset_name()
        else:
            raise ValueError("Invalid dataset_type. Must be 'training' or 'evaluation'.")
        return cls._path_from_data(dataset_name)


    # Modified dataset_preprocessed_dir_path to work with specific datasets
    @classmethod
    def dataset_preprocessed_dir_path(cls, dataset_name: str):
        # Get the base data path for the specific dataset
        base_data_path = cls._path_from_data(dataset_name)
        # Get the preprocessed directory relative to the dataset path
        preprocessed_dir = cls._dict["dataset_specific"][dataset_name]["preprocessed_dir"]
        return os.path.join(base_data_path, preprocessed_dir)


    # Modified dataset_emotions to work with specific datasets
    @classmethod
    def dataset_emotions(cls, dataset_name: str):
        return cls._dict["dataset_specific"][dataset_name]["emotions"]


    @classmethod
    def dataloader_dict(cls):
        return cls._dict["dataloader"]

    @classmethod
    def saved_models_location(cls):
        save_path = os.path.join(cls.project_root(), cls._dict["models"]["save_location"])
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        return save_path

    @classmethod
    def saved_ckpt_location(cls):
        save_path = os.path.join(cls.project_root(), cls._dict["models"]["checkpoints_location"])
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        return save_path

    @classmethod
    def pretrained_alexnet_url(cls):
        # Added check for existence as it might not be needed for this project
        return cls._dict["models"].get("pretrained_alexnet")

    @classmethod
    def save_plots_location(cls):
        save_path = os.path.join(cls.project_root(), cls._dict["plots"]["save_location"])
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        return save_path

    @classmethod
    def save_tables_location(cls):
        save_path = os.path.join(cls.project_root(), cls._dict["table"]["save_location"])
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        return save_path

    # New methods for model configurations
    @classmethod
    def audio_encoder_name(cls):
        return cls._dict["models"]["audio_encoder"]

    @classmethod
    def text_encoder_name(cls):
        return cls._dict["models"]["text_encoder"]

    @classmethod
    def fusion_dim(cls):
        """获取交叉注意力模块的工作维度"""
        # 从 models 配置中获取 fusion_dim，如果不存在则提供一个安全的默认值
        return cls._dict.get("models", {}).get("fusion_dim", 768)

    @classmethod
    def fusion_heads(cls):
        """获取交叉注意力模块的头数"""
        return cls._dict.get("models", {}).get("fusion_heads", 8)
    
    @classmethod
    def projection_bridge_config(cls):
        return cls._dict["models"]["projection_bridge"]

    @classmethod
    def training_epochs(cls):
        return cls._dict["training"]["epochs"]

    @classmethod
    def learning_rate(cls):
        return cls._dict["training"]["learning_rate"]
    
    @classmethod
    def training_head_lr(cls):
        """获取新添加的“头”部的学习率"""
        return float(cls._dict["training"]["head_lr"])
    
    @classmethod
    def weight_decay(cls):
        """获取权重衰减参数"""
        return float(cls._dict["training"]["weight_decay"])

    @classmethod
    def optimizer_type(cls):
        return cls._dict["training"]["optimizer"]

    @classmethod
    def llgca_loss_alpha(cls):
        return cls._dict["training"]["loss_weights"]["alpha"]

