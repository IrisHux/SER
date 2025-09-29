# select_best_checkpoint.py
import torch
import glob
import os
import pandas as pd
# ... 导入您的所有必要模块 ...

from core.config import CONFIG, device
from contrastive.trainer import ContrastiveTrainer
from contrastive.model import ContrastiveModel, MemoryOptimizedContrastiveModel
from scripts.get_dataloaders import get_contrastive_dataloaders

def main_selection():
    # --- 1. 初始化 ---
    CONFIG.load_config("config.yaml")
    checkpoint_dir = CONFIG.saved_models_location() # 从config获取保存路径
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.pth')) # 查找所有模型文件
    checkpoint_files.sort() # 排序

    results_list = []

    # --- 2. 准备验证集数据加载器 ---
    dataloaders = get_contrastive_dataloaders(CONFIG.training_dataset_name())
    validation_loader = dataloaders['validation']

    # --- 3. 循环评估每个检查点 ---
    for checkpoint_path in checkpoint_files:
        print(f"\n--- 正在验证检查点: {os.path.basename(checkpoint_path)} ---")

        # 实例化模型并加载权重
        num_labels = len(CONFIG.dataset_emotions(CONFIG.training_dataset_name()))
        model = MemoryOptimizedContrastiveModel(num_labels=num_labels).to(device)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # 创建一个临时评估器
        evaluator = ContrastiveTrainer(model=model, num_epochs=1, learning_rate=1e-5, alpha=2.0)

        # *** 关键：在 validation_loader 上进行评估 ***
        uar, war = evaluator.eval(validation_loader, labels=CONFIG.dataset_emotions(CONFIG.training_dataset_name()))

        results_list.append({'checkpoint': os.path.basename(checkpoint_path), 'uar': uar, 'war': war})

    # --- 4. 分析结果 ---
    results_df = pd.DataFrame(results_list)
    print("\n--- 所有检查点在验证集上的性能表现 ---")
    print(results_df.sort_values(by='uar', ascending=False))

    # 找到最佳检查点
    best_checkpoint_info = results_df.loc[results_df['uar'].idxmax()]
    best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint_info['checkpoint'])

    print(f"\n[结论] 最佳检查点为: {best_checkpoint_info['checkpoint']}")
    print(f"其在验证集上的UAR为: {best_checkpoint_info['uar']:.4f}")
    print(f"完整路径: {best_checkpoint_path}")

    return best_checkpoint_path

if __name__ == '__main__':
    best_model_path = main_selection()
    # 可以将最佳路径保存到一个文件中，供下一步使用