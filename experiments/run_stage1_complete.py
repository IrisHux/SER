"""
完整的 Stage 1 实验流程

包含：
- Step 1.1: 纯音频基线的超参数调优
- Step 1.2: 简单双模态融合的性能评估

运行方式：
python experiments/run_stage1_complete.py --step all
python experiments/run_stage1_complete.py --step 1.1  # 只运行音频基线调优
python experiments/run_stage1_complete.py --step 1.2  # 只运行双模态评估
"""

import os
import sys
import argparse
import json
import torch
import gc
import warnings
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import CONFIG, device
from scripts.get_dataloaders import get_dataloaders, get_contrastive_dataloaders
from audio.baseline_model import AudioBaselineModel
from audio.trainer import MemoryOptimizedAudioBaselineTrainer
from experiments.simple_multimodal_model import SimpleMultimodalModel, SimpleMultimodalTrainer

warnings.filterwarnings("ignore")


class Stage1CompleteExperiment:
    """完整的 Stage 1 实验管理器"""
    
    def __init__(self, output_dir: str = "experiments/results/stage1_complete"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        CONFIG.load_config("config.yaml")
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 数据集信息
        self.training_dataset_name = CONFIG.training_dataset_name()
        self.evaluation_dataset_name = CONFIG.evaluation_dataset_name()
        self.num_labels = len(CONFIG.dataset_emotions(self.training_dataset_name))
        
        # 加载数据
        print("=" * 80)
        print("加载数据集...")
        print("=" * 80)
        self._load_data()
        
        # 结果记录
        self.results = {
            'step1_1': [],  # 音频基线结果
            'step1_2': []   # 双模态结果
        }
    
    def _load_data(self):
        """加载所有需要的数据集"""
        print(f"加载 {self.training_dataset_name}...")
        # 对于音频基线，使用普通的dataloader
        train_loaders = get_dataloaders(self.training_dataset_name)
        self.train_loader_audio = train_loaders['train']
        self.val_loader_audio = train_loaders['validation']
        
        # 对于双模态，使用对比学习的dataloader（包含文本）
        multimodal_loaders = get_contrastive_dataloaders(self.training_dataset_name)
        self.train_loader = multimodal_loaders['train']
        self.val_loader = multimodal_loaders['validation']
        
        print(f"加载 {self.evaluation_dataset_name}...")
        eval_loaders = get_dataloaders(self.evaluation_dataset_name)
        self.test_loader_audio = eval_loaders['evaluation']
        
        multimodal_eval_loaders = get_contrastive_dataloaders(self.evaluation_dataset_name)
        self.test_loader = multimodal_eval_loaders['evaluation']
    
    def _cleanup(self):
        """清理GPU内存"""
        gc.collect()
        torch.cuda.empty_cache()
    
    # ==================== Step 1.1: 音频基线调优 ====================
    
    def step1_1_audio_baseline_tuning(self):
        """
        Step 1.1: 纯音频基线的超参数网格搜索
        
        调优的超参数:
        - learning_rate: [1e-5, 2e-5, 5e-5]
        - weight_decay: [0.0, 0.01, 0.05]
        """
        print("\n" + "=" * 80)
        print("Step 1.1: 音频基线超参数调优")
        print("=" * 80)
        
        # 超参数网格        
        lr_values = [1e-5, 2e-5, 5e-5]
        wd_values = [0.0, 0.01, 0.05]
        
        best_uar = 0
        best_config = None
        
        # 网格搜索
        for lr in lr_values:
            for wd in wd_values:
                exp_name = f"audio_lr{lr}_wd{wd}"
                print(f"\n实验: {exp_name}")
                
                result = self._train_audio_baseline(
                    learning_rate=lr,
                    weight_decay=wd,
                    num_epochs=10,
                    experiment_name=exp_name
                )
                
                self.results['step1_1'].append(result)
                
                # 更新最佳配置
                if result['val_uar'] > best_uar:
                    best_uar = result['val_uar']
                    best_config = {
                        'learning_rate': lr,
                        'weight_decay': wd,
                        'val_uar': result['val_uar'],
                        'test_uar': result['test_uar']
                    }
                
                # 保存中间结果
                self._save_step_results('step1_1')
        
        # 保存最佳配置
        best_config_path = os.path.join(self.output_dir, 'step1_1_best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print("\n" + "=" * 80)
        print("Step 1.1 完成！")
        print(f"最佳配置: LR={best_config['learning_rate']}, WD={best_config['weight_decay']}")
        print(f"验证集 UAR: {best_config['val_uar']:.4f}")
        print(f"测试集 UAR: {best_config['test_uar']:.4f}")
        print("=" * 80)
        
        return best_config
    
    def _train_audio_baseline(
        self,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        experiment_name: str
    ):
        """训练单个音频基线模型"""
        self._cleanup()
        
        # 创建模型
        model = AudioBaselineModel(num_labels=self.num_labels).to(device)
        
        # 创建训练器
        trainer = MemoryOptimizedAudioBaselineTrainer(
            model=model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            optimizer_type="AdamW",
            gradient_accumulation_steps=4
        )
        
        # 应用权重衰减
        if weight_decay > 0:
            trainer._optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 训练
        print(f"开始训练 {experiment_name}...")
        trainer.train(self.train_loader_audio, self.val_loader_audio)
        
        # 评估
        print("评估中...")
        val_uar, val_war, _ = trainer.eval(
            self.val_loader_audio,
            labels=CONFIG.dataset_emotions(self.training_dataset_name)
        )
        
        test_uar, test_war, _ = trainer.eval(
            self.test_loader_audio,
            labels=CONFIG.dataset_emotions(self.evaluation_dataset_name)
        )
        
        # 收集结果
        result = {
            'experiment_name': experiment_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'val_uar': val_uar,
            'val_war': val_war,
            'test_uar': test_uar,
            'test_war': test_war,
            'timestamp': datetime.now().isoformat()
        }
        
        # 清理
        del model, trainer
        self._cleanup()
        
        return result
    
    # ==================== Step 1.2: 简单双模态融合 ====================
    
    def step1_2_simple_multimodal(self, best_audio_config: dict):
        """
        Step 1.2: 简单双模态融合评估
        
        测试三种融合策略:
        - concat: 简单拼接
        - weighted_avg: 加权平均
        - gated: 门控融合
        
        Args:
            best_audio_config: Step 1.1中找到的最佳音频配置
        """
        print("\n" + "=" * 80)
        print("Step 1.2: 简单双模态融合评估")
        print("=" * 80)
        
        # 使用Step 1.1的最佳超参数
        lr = best_audio_config['learning_rate']
        wd = best_audio_config['weight_decay']
        
        # 测试不同的融合策略
        fusion_types = ['concat', 'weighted_avg', 'gated']
        
        best_uar = 0
        best_fusion = None
        
        for fusion_type in fusion_types:
            exp_name = f"multimodal_{fusion_type}"
            print(f"\n实验: {exp_name}")
            
            result = self._train_multimodal(
                fusion_type=fusion_type,
                learning_rate=lr,
                weight_decay=wd,
                num_epochs=10,
                experiment_name=exp_name
            )
            
            self.results['step1_2'].append(result)
            
            # 更新最佳融合策略
            if result['val_uar'] > best_uar:
                best_uar = result['val_uar']
                best_fusion = {
                    'fusion_type': fusion_type,
                    'learning_rate': lr,
                    'weight_decay': wd,
                    'val_uar': result['val_uar'],
                    'test_uar': result['test_uar']
                }
            
            # 保存中间结果
            self._save_step_results('step1_2')
        
        # 保存最佳配置
        best_config_path = os.path.join(self.output_dir, 'step1_2_best_config.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_fusion, f, indent=2)
        
        # 计算相对于音频基线的提升
        audio_baseline_uar = best_audio_config['val_uar']
        improvement = (best_uar - audio_baseline_uar) / audio_baseline_uar * 100
        
        print("\n" + "=" * 80)
        print("Step 1.2 完成！")
        print(f"最佳融合策略: {best_fusion['fusion_type']}")
        print(f"验证集 UAR: {best_fusion['val_uar']:.4f}")
        print(f"测试集 UAR: {best_fusion['test_uar']:.4f}")
        print(f"相对音频基线提升: {improvement:.2f}%")
        print("=" * 80)
        
        return best_fusion
    
    def _train_multimodal(
        self,
        fusion_type: str,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int,
        experiment_name: str
    ):
        """训练单个双模态模型"""
        self._cleanup()
        
        # 创建模型
        model = SimpleMultimodalModel(
            num_labels=self.num_labels,
            fusion_type=fusion_type,
            freeze_encoders=False  # 允许微调编码器
        )
        
        # 创建训练器
        trainer = SimpleMultimodalTrainer(
            model=model,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_accumulation_steps=4
        )
        
        # 训练
        print(f"开始训练 {experiment_name}...")
        trainer.train(self.train_loader, self.val_loader)
        
        # 评估
        print("评估中...")
        val_uar, val_war = trainer.evaluate(self.val_loader)
        test_uar, test_war = trainer.evaluate(self.test_loader)
        
        # 收集结果
        result = {
            'experiment_name': experiment_name,
            'fusion_type': fusion_type,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs,
            'val_uar': val_uar,
            'val_war': val_war,
            'test_uar': test_uar,
            'test_war': test_war,
            'timestamp': datetime.now().isoformat()
        }
        
        # 清理
        del model, trainer
        self._cleanup()
        
        return result
    
    # ==================== 结果保存 ====================
    
    def _save_step_results(self, step_name: str):
        """保存某个步骤的结果"""
        # JSON格式
        json_path = os.path.join(self.output_dir, f'{step_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results[step_name], f, indent=2)
        
        # CSV格式
        csv_path = os.path.join(self.output_dir, f'{step_name}_results.csv')
        df = pd.DataFrame(self.results[step_name])
        df.to_csv(csv_path, index=False)
    
    def save_all_results(self):
        """保存所有结果"""
        for step_name in ['step1_1', 'step1_2']:
            if self.results[step_name]:
                self._save_step_results(step_name)
    
    # ==================== 主运行流程 ====================
    
    def run_complete_stage1(self):
        """运行完整的Stage 1流程"""
        print("\n" + "=" * 80)
        print("开始完整的 Stage 1 实验")
        print("=" * 80)
        
        # Step 1.1: 音频基线调优
        best_audio_config = self.step1_1_audio_baseline_tuning()
        
        # Step 1.2: 双模态融合
        best_multimodal_config = self.step1_2_simple_multimodal(best_audio_config)
        
        # 生成总结报告
        self._generate_summary_report(best_audio_config, best_multimodal_config)
        
        print("\n" + "=" * 80)
        print("Stage 1 完整实验结束！")
        print(f"所有结果已保存到: {self.output_dir}")
        print("=" * 80)
    
    def _generate_summary_report(self, audio_config, multimodal_config):
        """生成总结报告"""
        report = {
            'stage': 'Stage 1: 基线建立与超参数初探',
            'step1_1': {
                'description': '音频基线超参数调优',
                'best_config': audio_config
            },
            'step1_2': {
                'description': '简单双模态融合',
                'best_config': multimodal_config,
                'improvement_over_audio': (
                    (multimodal_config['val_uar'] - audio_config['val_uar']) / 
                    audio_config['val_uar'] * 100
                )
            },
            'next_steps': [
                'Stage 2: 引入对比学习机制',
                'Stage 3: 测试不同的融合策略（交叉注意力等）',
                'Stage 4: 添加XBM等高级组件'
            ]
        }
        
        report_path = os.path.join(self.output_dir, 'stage1_summary_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印报告
        print("\n" + "=" * 80)
        print("Stage 1 总结报告")
        print("=" * 80)
        print(f"\nStep 1.1 最佳音频基线:")
        print(f"  - Learning Rate: {audio_config['learning_rate']}")
        print(f"  - Weight Decay: {audio_config['weight_decay']}")
        print(f"  - 验证集 UAR: {audio_config['val_uar']:.4f}")
        print(f"  - 测试集 UAR: {audio_config['test_uar']:.4f}")
        
        print(f"\nStep 1.2 最佳双模态融合:")
        print(f"  - Fusion Type: {multimodal_config['fusion_type']}")
        print(f"  - 验证集 UAR: {multimodal_config['val_uar']:.4f}")
        print(f"  - 测试集 UAR: {multimodal_config['test_uar']:.4f}")
        print(f"  - 相对提升: {report['step1_2']['improvement_over_audio']:.2f}%")
        
        print("\n下一步建议:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Stage 1 完整实验')
    parser.add_argument(
        '--step',
        type=str,
        default='all',
        choices=['all', '1.1', '1.2'],
        help='要运行的步骤 (all/1.1/1.2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results/stage1_complete',
        help='结果输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建实验管理器
    experiment = Stage1CompleteExperiment(output_dir=args.output_dir)
    
    if args.step == 'all':
        # 运行完整流程
        experiment.run_complete_stage1()
    elif args.step == '1.1':
        # 只运行Step 1.1
        best_config = experiment.step1_1_audio_baseline_tuning()
        experiment.save_all_results()
    elif args.step == '1.2':
        # 只运行Step 1.2（需要先有Step 1.1的结果）
        config_path = os.path.join(args.output_dir, 'step1_1_best_config.json')
        if not os.path.exists(config_path):
            print(f"错误: 未找到 {config_path}")
            print("请先运行 Step 1.1 或使用 --step all")
            return
        
        with open(config_path, 'r') as f:
            best_audio_config = json.load(f)
        
        experiment.step1_2_simple_multimodal(best_audio_config)
        experiment.save_all_results()


if __name__ == "__main__":
    main()
