"""
阶段1：建立基线并初步确定超参数范围

策略：
1. 先从最简单的模型开始（Audio-Only）
2. 在这个简单模型上调优基础超参数（lr, batch_size, weight_decay, warmup_steps）
3. 将这些超参数作为后续实验的默认值

使用方法:
python experiments/stage1_hyperparameter_tuning.py
"""

import os
import sys
import torch
import gc
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CONFIG, device
from scripts.get_dataloaders import get_dataloaders
from audio.baseline_model import AudioBaselineModel
from audio.trainer import MemoryOptimizedAudioBaselineTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Stage1Experiment:
    """
    阶段1实验管理器：系统化的超参数调优
    """
    
    def __init__(self, output_dir: str = "experiments/results/stage1"):
        """
        初始化实验管理器
        
        Args:
            output_dir: 实验结果保存目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        CONFIG.load_config("config.yaml")
        
        # 设置CUDA内存优化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 获取数据集信息
        self.training_dataset_name = CONFIG.training_dataset_name()
        self.evaluation_dataset_name = CONFIG.evaluation_dataset_name()
        self.num_labels = len(CONFIG.dataset_emotions(self.training_dataset_name))
        
        # 加载数据集（一次性加载，避免重复）
        print("=" * 80)
        print("加载数据集...")
        print("=" * 80)
        self.train_loader, self.val_loader = self._load_training_data()
        self.eval_loader = self._load_evaluation_data()
        
        # 实验结果记录
        self.results = []
        
    def _load_training_data(self):
        """加载训练和验证数据"""
        print(f"正在加载 {self.training_dataset_name} 数据集...")
        loaders = get_dataloaders(self.training_dataset_name)
        return loaders['train'], loaders['validation']
    
    def _load_evaluation_data(self):
        """加载评估数据"""
        print(f"正在加载 {self.evaluation_dataset_name} 数据集...")
        loaders = get_dataloaders(self.evaluation_dataset_name)
        return loaders['evaluation']
    
    def _cleanup_memory(self):
        """清理GPU内存"""
        gc.collect()
        torch.cuda.empty_cache()
    
    def _train_and_evaluate(
        self,
        learning_rate: float,
        batch_size: int = None,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.1,
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 4,
        experiment_name: str = "experiment"
    ) -> Dict:
        """
        训练并评估单个配置
        
        Returns:
            包含所有指标的字典
        """
        print("\n" + "=" * 80)
        print(f"实验: {experiment_name}")
        print("=" * 80)
        print(f"学习率: {learning_rate}")
        print(f"权重衰减: {weight_decay}")
        print(f"预热比例: {warmup_ratio}")
        print(f"训练轮数: {num_epochs}")
        print(f"梯度累积步数: {gradient_accumulation_steps}")
        print("=" * 80)
        
        # 清理内存
        self._cleanup_memory()
        
        # 创建模型
        model = AudioBaselineModel(num_labels=self.num_labels).to(device)
        
        # 创建训练器（带权重衰减支持）
        trainer = MemoryOptimizedAudioBaselineTrainer(
            model=model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type="AdamW",  # 使用AdamW以支持weight_decay
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        # 如果需要weight_decay，重新创建optimizer
        if weight_decay > 0:
            trainer._optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 训练模型
        print("\n开始训练...")
        training_history = trainer.train(self.train_loader, self.val_loader)
        
        # 在验证集上评估
        print("\n在验证集上评估...")
        val_uar, val_war, _ = trainer.eval(
            self.val_loader,
            labels=CONFIG.dataset_emotions(self.training_dataset_name)
        )
        
        # 在测试集上评估（零样本）
        print("\n在测试集上评估（零样本）...")
        test_uar, test_war, _ = trainer.eval(
            self.eval_loader,
            labels=CONFIG.dataset_emotions(self.evaluation_dataset_name)
        )
        
        # 收集结果
        result = {
            'experiment_name': experiment_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'num_epochs': num_epochs,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'val_uar': val_uar,
            'val_war': val_war,
            'test_uar': test_uar,
            'test_war': test_war,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # 清理模型
        del model, trainer
        self._cleanup_memory()
        
        return result
    
    def step1_1_tune_learning_rate(self):
        """
        Step 1.1: 调优学习率
        
        测试不同的学习率，找到最优值
        """
        print("\n" + "=" * 80)
        print("Step 1.1: 学习率调优")
        print("=" * 80)
        
        # 学习率候选值（对于WavLM微调，通常在1e-5到5e-5之间）
        lr_candidates = [1e-5, 2e-5, 3e-5, 5e-5]
        
        step1_1_results = []
        
        for lr in lr_candidates:
            result = self._train_and_evaluate(
                learning_rate=lr,
                weight_decay=0.01,  # 使用默认的小量权重衰减
                warmup_ratio=0.1,
                num_epochs=10,
                gradient_accumulation_steps=4,
                experiment_name=f"step1.1_lr_{lr}"
            )
            step1_1_results.append(result)
            self.results.append(result)
            
            # 保存中间结果
            self._save_results()
        
        # 找到最优学习率
        best_result = max(step1_1_results, key=lambda x: x['val_uar'])
        best_lr = best_result['learning_rate']
        
        print("\n" + "=" * 80)
        print(f"Step 1.1 完成！最优学习率: {best_lr}")
        print(f"验证集 UAR: {best_result['val_uar']:.4f}")
        print(f"测试集 UAR: {best_result['test_uar']:.4f}")
        print("=" * 80)
        
        return best_lr, step1_1_results
    
    def step1_2_tune_weight_decay(self, best_lr: float):
        """
        Step 1.2: 调优权重衰减
        
        Args:
            best_lr: Step 1.1中找到的最优学习率
        """
        print("\n" + "=" * 80)
        print("Step 1.2: 权重衰减调优")
        print("=" * 80)
        
        # 权重衰减候选值
        wd_candidates = [0.0, 0.01, 0.05, 0.1]
        
        step1_2_results = []
        
        for wd in wd_candidates:
            result = self._train_and_evaluate(
                learning_rate=best_lr,
                weight_decay=wd,
                warmup_ratio=0.1,
                num_epochs=10,
                gradient_accumulation_steps=4,
                experiment_name=f"step1.2_wd_{wd}"
            )
            step1_2_results.append(result)
            self.results.append(result)
            
            # 保存中间结果
            self._save_results()
        
        # 找到最优权重衰减
        best_result = max(step1_2_results, key=lambda x: x['val_uar'])
        best_wd = best_result['weight_decay']
        
        print("\n" + "=" * 80)
        print(f"Step 1.2 完成！最优权重衰减: {best_wd}")
        print(f"验证集 UAR: {best_result['val_uar']:.4f}")
        print(f"测试集 UAR: {best_result['test_uar']:.4f}")
        print("=" * 80)
        
        return best_wd, step1_2_results
    
    def step1_3_tune_warmup(self, best_lr: float, best_wd: float):
        """
        Step 1.3: 调优预热步数
        
        Args:
            best_lr: 最优学习率
            best_wd: 最优权重衰减
        """
        print("\n" + "=" * 80)
        print("Step 1.3: 预热比例调优")
        print("=" * 80)
        
        # 预热比例候选值
        warmup_candidates = [0.0, 0.05, 0.1, 0.15]
        
        step1_3_results = []
        
        for warmup in warmup_candidates:
            result = self._train_and_evaluate(
                learning_rate=best_lr,
                weight_decay=best_wd,
                warmup_ratio=warmup,
                num_epochs=10,
                gradient_accumulation_steps=4,
                experiment_name=f"step1.3_warmup_{warmup}"
            )
            step1_3_results.append(result)
            self.results.append(result)
            
            # 保存中间结果
            self._save_results()
        
        # 找到最优预热比例
        best_result = max(step1_3_results, key=lambda x: x['val_uar'])
        best_warmup = best_result['warmup_ratio']
        
        print("\n" + "=" * 80)
        print(f"Step 1.3 完成！最优预热比例: {best_warmup}")
        print(f"验证集 UAR: {best_result['val_uar']:.4f}")
        print(f"测试集 UAR: {best_result['test_uar']:.4f}")
        print("=" * 80)
        
        return best_warmup, step1_3_results
    
    def _save_results(self):
        """保存实验结果到JSON和CSV"""
        # 保存为JSON（包含完整信息）
        json_path = os.path.join(self.output_dir, "results_full.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV（用于快速查看）
        csv_path = os.path.join(self.output_dir, "results_summary.csv")
        df = pd.DataFrame([
            {
                'experiment': r['experiment_name'],
                'lr': r['learning_rate'],
                'wd': r['weight_decay'],
                'warmup': r['warmup_ratio'],
                'val_uar': r['val_uar'],
                'val_war': r['val_war'],
                'test_uar': r['test_uar'],
                'test_war': r['test_war'],
            }
            for r in self.results
        ])
        df.to_csv(csv_path, index=False)
        
        print(f"\n结果已保存到: {self.output_dir}")
    
    def run_full_stage1(self):
        """
        运行完整的Stage 1实验流程
        """
        print("\n" + "=" * 80)
        print("开始 Stage 1: 超参数调优")
        print("=" * 80)
        
        # Step 1.1: 调优学习率
        best_lr, _ = self.step1_1_tune_learning_rate()
        
        # Step 1.2: 调优权重衰减
        best_wd, _ = self.step1_2_tune_weight_decay(best_lr)
        
        # Step 1.3: 调优预热比例
        best_warmup, _ = self.step1_3_tune_warmup(best_lr, best_wd)
        
        # 保存最优超参数
        best_params = {
            'learning_rate': best_lr,
            'weight_decay': best_wd,
            'warmup_ratio': best_warmup,
            'gradient_accumulation_steps': 4,
            'num_epochs': 10
        }
        
        params_path = os.path.join(self.output_dir, "best_hyperparameters.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print("\n" + "=" * 80)
        print("Stage 1 完成！")
        print("=" * 80)
        print("最优超参数:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\n结果保存在: {self.output_dir}")
        print("=" * 80)
        
        return best_params


def main():
    """主函数"""
    # 创建实验管理器
    experiment = Stage1Experiment(output_dir="experiments/results/stage1_audio_only")
    
    # 运行完整的Stage 1实验
    best_params = experiment.run_full_stage1()
    
    print("\n实验完成！")
    print(f"最优超参数已保存，可用于后续实验。")


if __name__ == "__main__":
    main()
