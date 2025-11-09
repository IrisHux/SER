# run_ablation.py

"""
统一的消融实验运行脚本（训练 + 评估）
支持运行两种消融实验：
1. Ablation A: LGCA w/o Label-Guidance (无标签监督)
2. Ablation B: LGCA w/o Text Anchor (无文本锚点)

类似于 run_contrastive.py 的结构，但用于消融实验
"""

import torch
import gc
import os
import logging
import numpy as np
import random
import warnings
import argparse
from pathlib import Path

# 导入项目核心模块
from core.config import CONFIG, device
from contrastive.model import setup_memory_optimization, MemoryOptimizedContrastiveModel, AcousticSupConModel
from contrastive.trainer import AblationNoLabelTrainer, AblationNoTextTrainer
from scripts.get_dataloaders import get_contrastive_dataloaders, get_ablation_no_text_dataloaders
from scripts.contrastive_ops import ModelOps  # 使用通用工具类
from vizualisers.plots import PlotVisualizer  # 修复：导入 PlotVisualizer 类

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_environment():
    """准备实验环境：加载配置、设置随机种子、优化内存"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # 1. 加载配置
    try:
        CONFIG.load_config("config.yaml")
        logger.info("配置文件 'config.yaml' 加载成功。")
    except FileNotFoundError:
        logger.error("错误：找不到 'config.yaml' 文件。")
        raise
    
    # 2. 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 3. 内存优化
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    setup_memory_optimization()
    torch.cuda.empty_cache()
    gc.collect()
    
    logger.info("环境准备完成。")


def run_ablation_no_label(run_training: bool = True, run_evaluation: bool = True, alpha: float = None):
    """
    运行消融实验 A: LGCA w/o Label-Guidance (训练 + 评估)
    
    Args:
        run_training: 是否运行训练
        run_evaluation: 是否运行评估
        alpha: 损失权重系数 (如果为 None，则从 config.yaml 读取)
    """
    logger.info("\n" + "="*80)
    logger.info("消融实验 A: LGCA w/o Label-Guidance")
    logger.info("="*80)
    
    # 从 config 获取 alpha 值
    if alpha is None:
        alpha = CONFIG.llgca_loss_alpha()
        logger.info(f"从配置文件读取 alpha = {alpha}")
    
    training_dataset_name = CONFIG.training_dataset_name()
    evaluation_dataset_name = CONFIG.evaluation_dataset_name()
    num_labels = len(CONFIG.dataset_emotions(training_dataset_name))
    
    # === 阶段 1: 训练 ===
    if run_training:
        logger.info("\n==================== [阶段 1: 开始训练] ====================")
        
        # 1. 创建模型（使用 ModelOps）
        model = ModelOps.create_or_load_model(
            model_class=MemoryOptimizedContrastiveModel,
            num_labels=num_labels
        )
        
        # 2. 创建训练器（使用 ModelOps）
        trainer = ModelOps.create_trainer(
            trainer_class=AblationNoLabelTrainer,
            model=model,
            alpha=alpha
        )
        
        # 3. 运行训练（使用 ModelOps）
        ModelOps.train(
            trainer=trainer,
            training_dataset_name=training_dataset_name,
            dataloader_func=get_contrastive_dataloaders
        )
        
        # 4. 训练后立即在验证集上评估
        logger.info("--- 对训练完成的模型在验证集上进行评估 ---")
        ModelOps.evaluate(
            trainer=trainer,
            dataset_split='validation',
            dataset_name=training_dataset_name,
            dataloader_func=get_contrastive_dataloaders
        )
        
        logger.info("==================== [阶段 1: 训练完成] ====================\n")
        
        # 清理内存
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
    
    # === 阶段 2: 评估所有检查点 ===
    if run_evaluation:
        logger.info("\n==================== [阶段 2: 评估所有检查点] ====================")
        
        # 使用 ModelOps 批量评估
        df_results, best_conf_matrix, test_emotions = ModelOps.evaluate_all_checkpoints(
            model_class=MemoryOptimizedContrastiveModel,
            trainer_class=AblationNoLabelTrainer,
            checkpoint_pattern='Ablation_LGCA_no_Label_model_epoch_*.pt',
            training_dataset_name=training_dataset_name,
            evaluation_dataset_name=evaluation_dataset_name,
            dataloader_func=get_contrastive_dataloaders,
            alpha=alpha
        )
        
        if not df_results.empty:
            # 保存结果到CSV
            results_path = Path(CONFIG.save_tables_location()) / f"ablation_no_label_evaluation_results_alpha{alpha}.csv"
            df_results.to_csv(results_path, index=False)
            logger.info(f"\n结果已保存到: {results_path}")
            
            # 保存最佳混淆矩阵
            if best_conf_matrix is not None:
                best_checkpoint = df_results.iloc[df_results['test_uar'].idxmax()]['checkpoint']
                best_uar = df_results['test_uar'].max()
                
                PlotVisualizer.plot_confusion_matrix(
                    best_conf_matrix,
                    test_emotions,
                    filename=f"ablation_no_label_best_model_cm.png"
                )
                
                logger.info(f"\n🏆 最佳模型检查点: {best_checkpoint}")
                logger.info(f"   最佳测试集 UAR: {best_uar:.2f}%")
                logger.info("   混淆矩阵已保存")
        
        logger.info("==================== [阶段 2: 评估完成] ====================\n")


def run_ablation_no_text(run_training: bool = True, run_evaluation: bool = True, alpha: float = None):
    """
    运行消融实验 B: LGCA w/o Text Anchor (训练 + 评估)
    
    Args:
        run_training: 是否运行训练
        run_evaluation: 是否运行评估
        alpha: 损失权重系数 (如果为 None，则从 config.yaml 读取)
    """
    logger.info("\n" + "="*80)
    logger.info("消融实验 B: LGCA w/o Text Anchor")
    logger.info("="*80)
    
    # 从 config 获取 alpha 值
    if alpha is None:
        alpha = CONFIG.llgca_loss_alpha()
        logger.info(f"从配置文件读取 alpha = {alpha}")
    
    training_dataset_name = CONFIG.training_dataset_name()
    evaluation_dataset_name = CONFIG.evaluation_dataset_name()
    num_labels = len(CONFIG.dataset_emotions(training_dataset_name))
    
    # === 阶段 1: 训练 ===
    if run_training:
        logger.info("\n==================== [阶段 1: 开始训练] ====================")
        
        # 1. 创建纯声学模型（使用 ModelOps）
        model = ModelOps.create_or_load_model(
            model_class=AcousticSupConModel,
            num_labels=num_labels
        )
        
        # 2. 创建训练器（使用 ModelOps）
        trainer = ModelOps.create_trainer(
            trainer_class=AblationNoTextTrainer,
            model=model,
            alpha=alpha
        )
        
        # 3. 运行训练（使用 ModelOps）
        ModelOps.train(
            trainer=trainer,
            training_dataset_name=training_dataset_name,
            dataloader_func=get_ablation_no_text_dataloaders  # 注意：使用特殊的数据加载器
        )
        
        # 4. 训练后立即在验证集上评估
        logger.info("--- 对训练完成的模型在验证集上进行评估 ---")
        ModelOps.evaluate(
            trainer=trainer,
            dataset_split='validation',
            dataset_name=training_dataset_name,
            dataloader_func=get_ablation_no_text_dataloaders
        )
        
        logger.info("==================== [阶段 1: 训练完成] ====================\n")
        
        # 清理内存
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()
    
    # === 阶段 2: 评估所有检查点 ===
    if run_evaluation:
        logger.info("\n==================== [阶段 2: 评估所有检查点] ====================")
        
        # 使用 ModelOps 批量评估
        df_results, best_conf_matrix, test_emotions = ModelOps.evaluate_all_checkpoints(
            model_class=AcousticSupConModel,
            trainer_class=AblationNoTextTrainer,
            checkpoint_pattern='Ablation_LGCA_no_Text_model_epoch_*.pt',
            training_dataset_name=training_dataset_name,
            evaluation_dataset_name=evaluation_dataset_name,
            dataloader_func=get_ablation_no_text_dataloaders,
            alpha=alpha
        )
        
        if not df_results.empty:
            # 保存结果到CSV
            results_path = Path(CONFIG.save_tables_location()) / f"ablation_no_text_evaluation_results_alpha{alpha}.csv"
            df_results.to_csv(results_path, index=False)
            logger.info(f"\n结果已保存到: {results_path}")
            
            # 保存最佳混淆矩阵
            if best_conf_matrix is not None:
                best_checkpoint = df_results.iloc[df_results['test_uar'].idxmax()]['checkpoint']
                best_uar = df_results['test_uar'].max()
                
                PlotVisualizer.plot_confusion_matrix(
                    best_conf_matrix,
                    test_emotions,
                    filename=f"ablation_no_text_best_model_cm.png"
                )
                
                logger.info(f"\n🏆 最佳模型检查点: {best_checkpoint}")
                logger.info(f"   最佳测试集 UAR: {best_uar:.2f}%")
                logger.info("   混淆矩阵已保存")
        
        logger.info("==================== [阶段 2: 评估完成] ====================\n")


def main():
    """主函数：解析参数并运行指定的消融实验"""
    parser = argparse.ArgumentParser(description='运行LGCA消融实验（训练+评估）')
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['no_label', 'no_text', 'both'],
        default='both',
        help='要运行的消融实验: no_label (无标签监督), no_text (无文本锚点), both (两者都运行)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='损失权重系数 alpha (可选，如果不指定则从 config.yaml 读取)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='只运行训练，不运行评估'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='只运行评估，不运行训练'
    )
    
    args = parser.parse_args()
    
    # 确定运行阶段
    run_training = not args.eval_only
    run_evaluation = not args.train_only
    
    # 准备环境
    prepare_environment()
    
    # 根据参数运行实验
    if args.experiment in ['no_label', 'both']:
        try:
            run_ablation_no_label(
                run_training=run_training,
                run_evaluation=run_evaluation,
                alpha=args.alpha
            )
        except Exception as e:
            logger.error(f"消融实验 A (无标签监督) 失败: {e}", exc_info=True)
    
    if args.experiment in ['no_text', 'both']:
        try:
            run_ablation_no_text(
                run_training=run_training,
                run_evaluation=run_evaluation,
                alpha=args.alpha
            )
        except Exception as e:
            logger.error(f"消融实验 B (无文本锚点) 失败: {e}", exc_info=True)
    
    logger.info("\n" + "="*80)
    logger.info("所有消融实验完成！")
    logger.info("="*80)


if __name__ == "__main__":
    main()
