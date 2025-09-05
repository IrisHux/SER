# evaluate_models.py

import torch
import argparse # 用于解析命令行参数
import os

# 导入您项目中的核心模块
from core.config import CONFIG, device
from scripts.get_dataloaders import get_contrastive_dataloaders
from audio.baseline_model import AudioBaselineModel # 导入基线模型
from contrastive.model import ContrastiveModel      # 导入LGCA模型
from contrastive.trainer import ContrastiveTrainer  # 我们将复用它的eval方法

def main():
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="标准化模型评估脚本")
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'lgca'],
                        help="要评估的模型类型: 'baseline' 或 'lgca'")
    parser.add_argument('--model_path', type=str, required=True,
                        help="已保存的模型权重文件 (.pth) 的路径")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['IEMOCAP', 'CREMA-D'],
                        help="用于评估的数据集名称: 'IEMOCAP' 或 'CREMA-D'")
    args = parser.parse_args()

    print(f"--- [开始评估] 模型类型: {args.model_type}, 数据集: {args.dataset_name} ---")

    # 2. 加载配置
    CONFIG.load_config("config.yaml")
    num_labels = len(CONFIG.dataset_emotions(args.dataset_name))

    # 3. 根据类型实例化对应的模型架构
    if args.model_type == 'baseline':
        # 注意：这里需要一个纯声学的Trainer来评估，因为它只需要音频输入
        # 如果您的ContrastiveTrainer可以处理纯音频评估，也可以复用
        # 为简化，我们假设ContrastiveTrainer的eval方法是通用的
        model = AudioBaselineModel(num_labels=num_labels).to(device)
    elif args.model_type == 'lgca':
        model = ContrastiveModel(num_labels=num_labels).to(device)
    else:
        raise ValueError("未知的模型类型")

    # 4. 加载已训练好的模型权重
    try:
        # 遵循最佳实践，使用 weights_only=True
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        model.eval() # 切换到评估模式
        print("模型权重加载成功。")
    except Exception as e:
        print(f"错误：模型权重加载失败。请检查路径或模型架构。错误: {e}")
        return

    # 5. 准备评估数据加载器
    print("准备评估数据...")
    if args.dataset_name == 'IEMOCAP':
        dataloaders = get_contrastive_dataloaders(CONFIG.training_dataset_name())
        eval_loader = dataloaders['validation'] # 在IEMOCAP上，我们用验证集进行评估
    else: # CREMA-D
        dataloaders = get_contrastive_dataloaders(CONFIG.evaluation_dataset_name())
        eval_loader = dataloaders['evaluation']

    # 6. 执行评估
    # 我们可以临时创建一个Trainer实例，只为了使用它标准化的eval方法
    # 注意：这里的lr, alpha等参数在评估时不会被用到
    evaluator = ContrastiveTrainer(
        model=model, num_epochs=1, learning_rate=1e-5, alpha=0.5
    )

    print("开始在目标数据集上进行评估...")
    evaluator.eval(eval_loader, labels=CONFIG.dataset_emotions(args.dataset_name))

    print(f"--- [评估完成] 模型类型: {args.model_type}, 数据集: {args.dataset_name} ---")

if __name__ == '__main__':
    main()