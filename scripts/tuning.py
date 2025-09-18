# tuning.py

import torch
import pandas as pd
import itertools
from torch.utils.data import DataLoader, Subset # 确保导入 Subset
import warnings # <-- 新增

warnings.filterwarnings("ignore", category=UserWarning)



# 导入所有必要的模块
from core.config import CONFIG, device
from dataloaders.dataset import EmotionDataset
from contrastive.collator import ContrastiveDataCollator
from contrastive.model import ContrastiveModel, MemoryOptimizedContrastiveModel
from contrastive.trainer import ContrastiveTrainer
# 确保您已经创建了这个新的数据加载器脚本
from scripts.get_dataloaders import get_contrastive_dataloaders


# --- 这是我们设计的最终函数 ---

def run_hyperparameter_trial(alpha_value: float,
                             train_loader: DataLoader,
                             validation_loader: DataLoader,
                             config: object,
                             num_epochs: int,
                             gradient_accumulation_steps: int):
    """
    执行一次完整的超参数实验。

    Args:
        alpha_value (float): 本次试验要使用的 alpha 值。
        train_loader (DataLoader): 训练数据加载器。
        validation_loader (DataLoader): 验证数据加载器。
        config (dict): 包含其他参数的配置对象 (例如 epochs, lr)。

    Returns:
        tuple: (uar, war) 在验证集上的性能指标。
    """
    print(f"\n--- [试验开始] Alpha: {alpha_value} ---")


    # 1. 初始化模型和优化器
    # 每次调用都创建一个全新的模型实例，确保实验的独立性
    num_labels = len(CONFIG.dataset_emotions(CONFIG.training_dataset_name()))
    # model = ContrastiveModel(num_labels=num_labels).to(device)
    model = MemoryOptimizedContrastiveModel(num_labels=num_labels).to(device)

    # 将 alpha_value 传递给训练器
    trainer = ContrastiveTrainer(
        model=model,
        # num_epochs=CONFIG.training_epochs(),
        # learning_rate=CONFIG.learning_rate(),
        # num_epochs=config.training_epochs(),      # <-- 现在会正确地调用 TempConfig 的方法，得到 1
        num_epochs = num_epochs,
        learning_rate=config.learning_rate(), # <-- 也使用传入的 config
        alpha=alpha_value,  # <--- 将参数传入
        # optimizer_type=CONFIG.optimizer_type(),
        optimizer_type=config.optimizer_type(),
        gradient_accumulation_steps=gradient_accumulation_steps # <-- 使用传入的参数
    )

    # 2. 运行完整的训练流程
    print("开始训练...")
    trainer.train(train_loader, validation_loader)
    print("训练完成。")

    # 3. 在IEMOCAP验证集上进行评估
    print("在IEMOCAP验证集上进行评估...")
    iemocap_emotions = CONFIG.dataset_emotions(CONFIG.training_dataset_name())
    # 确保您的 eval 方法返回 uar 和 war
    val_uar, val_war = trainer.eval(validation_loader, labels=iemocap_emotions)

    # 4. 返回验证集上的UAR和WAR指标
    print(f"试验完成: alpha={alpha_value}, UAR={val_uar:.4f}, WAR={val_war:.4f}")
    return val_uar, val_war

