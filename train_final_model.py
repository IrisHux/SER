# train_final_model.py

import torch
import os
import gc
import numpy as np

# 导入您项目中的核心模块
from core.config import CONFIG, device
from scripts.get_dataloaders import get_contrastive_dataloaders
from contrastive.model import ContrastiveModel
from contrastive.trainer import ContrastiveTrainer
from torch.utils.data import ConcatDataset # <-- 导入用于合并数据集的工具

def train_final_lgca_model():
    """
    使用最优超参数和全部训练数据，训练并保存最终的LGCA模型。
    """
    print("--- [最终模型训练开始] ---")

    # 1. 加载配置
    CONFIG.load_config("config.yaml")

    # 为可复现性设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. 加载并合并数据集, 最大化训练数据利用率
    print("加载并合并IEMOCAP训练集与验证集...")
    # 注意：get_contrastive_dataloaders 内部会进行 train/val 划分
    # 我们需要获取它划分出的两个数据集对象
    dataloaders = get_contrastive_dataloaders(CONFIG.training_dataset_name())
    train_dataset = dataloaders['train'].dataset
    validation_dataset = dataloaders['validation'].dataset

    # 使用 ConcatDataset 将训练集和验证集合并
    full_train_dataset = ConcatDataset([train_dataset, validation_dataset])

    # 用合并后的完整数据集创建一个新的 DataLoader
    full_train_loader = torch.utils.data.DataLoader(
        full_train_dataset,
        batch_size=CONFIG.dataloader_dict()['batch_size'],
        collate_fn=dataloaders['train'].collate_fn, # collate_fn 保持不变
        num_workers=CONFIG.dataloader_dict()['num_workers'],
        shuffle=True # 在最终训练时打乱数据
    )
    print(f"数据准备完成。总训练样本数: {len(full_train_dataset)}")

    # 3. 初始化模型和训练器
    print("初始化模型和训练器...")
    num_labels = len(CONFIG.dataset_emotions(CONFIG.training_dataset_name()))
    model = ContrastiveModel(num_labels=num_labels).to(device)

    trainer = ContrastiveTrainer(
        model=model,
        num_epochs=CONFIG.training_epochs(),
        learning_rate=CONFIG.learning_rate(),
        alpha=CONFIG.llgca_loss_alpha(), # 从config中读取最优alpha (应为2.0)
        optimizer_type=CONFIG.optimizer_type(),
        gradient_accumulation_steps=4
    )

    # 4. 执行训练
    print(f"使用最优 alpha = {trainer.alpha} 开始在完整数据上进行最终训练...")
    trainer.train(full_train_loader)
    print("最终训练完成。")

    # 5. 保存模型权重
    model_save_path = os.path.join(
        CONFIG.saved_models_location(), 
        f"lgca_final_alpha_{trainer.alpha}_seed_{seed}.pth"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已成功保存至: {model_save_path}")

# --- 脚本执行入口 ---
if __name__ == '__main__':
    train_final_lgca_model()