# contrastive_ops.py

"""
通用的模型操作工具类
提供模型创建、加载、训练、评估等通用功能
支持主模型和消融实验模型
"""

import torch
import gc
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Union, Tuple, Optional, List, Callable

# 导入您项目中的核心模块
from core.config import CONFIG, device
from contrastive.model import MemoryOptimizedContrastiveModel, AcousticSupConModel
from contrastive.trainer import ContrastiveTrainer, AblationNoLabelTrainer, AblationNoTextTrainer
from scripts.get_dataloaders import get_contrastive_dataloaders

logger = logging.getLogger(__name__)

class ModelOps:
    """
    通用的模型操作工具类
    封装模型创建、加载、训练和评估等通用操作
    支持主模型和消融实验模型
    """

    @classmethod
    def create_or_load_model(
        cls, 
        model_class: type,
        num_labels: int,
        checkpoint_path: Optional[str] = None
    ) -> Union[MemoryOptimizedContrastiveModel, AcousticSupConModel]:
        """
        创建新模型或从检查点加载模型（通用方法）
        
        Args:
            model_class: 模型类 (例如 MemoryOptimizedContrastiveModel 或 AcousticSupConModel)
            num_labels: 情感类别数量
            checkpoint_path: 检查点文件的完整路径 (可选)
        
        Returns:
            初始化（或加载）后的模型实例
        """
        # 创建模型实例
        model = model_class(num_labels=num_labels).to(device)
        
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                logger.error(f"错误：找不到检查点文件 {checkpoint_path}")
                raise FileNotFoundError(f"找不到检查点文件 {checkpoint_path}")
                
            logger.info(f"--- 正在从 {checkpoint_path} 加载模型权重 ---")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            logger.info(f"--- 正在创建新的 {model_class.__name__} ---")
            
        return model

    @classmethod
    def create_trainer(
        cls, 
        trainer_class: type,
        model: Union[MemoryOptimizedContrastiveModel, AcousticSupConModel],
        alpha: Optional[float] = None
    ) -> Union[ContrastiveTrainer, AblationNoLabelTrainer, AblationNoTextTrainer]:
        """
        根据配置创建训练器实例（通用方法）
        
        Args:
            trainer_class: 训练器类 (例如 ContrastiveTrainer, AblationNoLabelTrainer)
            model: 模型实例
            alpha: 损失权重系数 (可选，如果为 None 则从 config 读取)
        
        Returns:
            训练器实例
        """
        if alpha is None:
            alpha = CONFIG.llgca_loss_alpha()
            
        logger.info(f"--- 正在初始化 {trainer_class.__name__} (alpha={alpha}) ---")
        trainer = trainer_class(
            model=model,
            num_epochs=CONFIG.training_epochs(),
            learning_rate=CONFIG.learning_rate(),
            alpha=alpha,
            optimizer_type=CONFIG.optimizer_type(),
            gradient_accumulation_steps=4,
            # XBM配置
            use_xbm=True,           # 启用XBM
            xbm_memory_size=16384   # 记忆库大小（建议16384或更大）
        )
        return trainer

    @classmethod
    def train(
        cls,
        trainer: Union[ContrastiveTrainer, AblationNoLabelTrainer, AblationNoTextTrainer],
        training_dataset_name: str,
        dataloader_func: Callable = get_contrastive_dataloaders
    ):
        """
        通用的训练方法
        
        Args:
            trainer: 训练器实例
            training_dataset_name: 训练数据集名称
            dataloader_func: 数据加载器函数
        """
        logger.info(f"--- 正在为训练集 '{training_dataset_name}' 准备 Dataloaders ---")
        try:
            loaders = dataloader_func(training_dataset_name)
            train_loader = loaders['train']
            validation_loader = loaders['validation']
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return

        logger.info(f"--- 开始在 '{training_dataset_name}' 上进行训练，共 {trainer._num_epochs} 个 Epochs ---")
        trainer.train(train_loader, validation_loader) 
        logger.info("--- 训练完成 ---")

    @classmethod
    def evaluate(
        cls,
        trainer: Union[ContrastiveTrainer, AblationNoLabelTrainer, AblationNoTextTrainer],
        dataset_split: str,
        dataset_name: str,
        dataloader_func: Callable = get_contrastive_dataloaders
    ) -> Tuple[float, float, np.ndarray]:
        """
        通用的单次评估方法
        
        Args:
            trainer: 训练器实例
            dataset_split: 数据集拆分 ('train', 'validation', 'test', 'evaluation')
            dataset_name: 数据集名称
            dataloader_func: 数据加载器函数
            
        Returns:
            Tuple[float, float, np.ndarray]: (uar, war, confusion_matrix)
        """
        logger.info(f"--- 正在为 '{dataset_name}' 的 '{dataset_split}' 拆分准备 Dataloader ---")
        try:
            loaders = dataloader_func(dataset_name)
            loader = loaders[dataset_split]
            emotions = CONFIG.dataset_emotions(dataset_name)
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None, None, None

        logger.info(f"--- 正在 '{dataset_split}' 拆分上评估模型 ---")
        uar, war, conf_matrix = trainer.eval(loader)
        logger.info(f"评估结果 ({dataset_split}) - UAR: {uar:.2f}%, WAR: {war:.2f}%")
        return uar, war, conf_matrix

    @classmethod
    def evaluate_all_checkpoints(
        cls,
        model_class: type,
        trainer_class: type,
        checkpoint_pattern: str,
        training_dataset_name: str,
        evaluation_dataset_name: str,
        dataloader_func: Callable = get_contrastive_dataloaders,
        alpha: Optional[float] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        通用的检查点批量评估方法
        
        Args:
            model_class: 模型类 (例如 MemoryOptimizedContrastiveModel)
            trainer_class: 训练器类 (例如 ContrastiveTrainer)
            checkpoint_pattern: 检查点文件名匹配模式 (例如 'Contrastive_LGCA_model_epoch_*.pt')
            training_dataset_name: 训练数据集名称（用于获取类别数）
            evaluation_dataset_name: 评估数据集名称
            dataloader_func: 数据加载器函数 (默认 get_contrastive_dataloaders)
            alpha: 损失权重系数 (可选)
        
        Returns:
            Tuple[pd.DataFrame, np.ndarray, List[str]]: 
                (评估结果DataFrame, 最佳混淆矩阵, 情感标签列表)
        """
        logger.info("\n" + "="*80)
        logger.info(f"开始批量评估检查点: {checkpoint_pattern}")
        logger.info("="*80)
        
        # --- 1. 准备数据加载器 ---
        logger.info(f"正在为训练集 '{training_dataset_name}' 准备数据加载器...")
        try:
            train_loaders = dataloader_func(training_dataset_name)
            validation_loader = train_loaders['validation']
        except Exception as e:
            logger.error(f"训练数据加载失败: {e}")
            return pd.DataFrame(), None, []
        
        logger.info(f"正在为测试集 '{evaluation_dataset_name}' 准备数据加载器...")
        try:
            test_loaders = dataloader_func(evaluation_dataset_name)
            test_loader = test_loaders['test'] if 'test' in test_loaders else test_loaders['evaluation']
            test_emotions = CONFIG.dataset_emotions(evaluation_dataset_name)
        except Exception as e:
            logger.error(f"测试数据加载失败: {e}")
            return pd.DataFrame(), None, []

        # --- 2. 查找检查点文件 ---
        checkpoint_dir = Path(CONFIG.saved_ckpt_location())  # 修复：使用正确的检查点目录
        if not checkpoint_dir.exists():
            logger.error(f"错误：找不到检查点目录 '{checkpoint_dir}'")
            return pd.DataFrame(), None, []
        
        checkpoint_files = sorted(checkpoint_dir.glob(checkpoint_pattern))
        if not checkpoint_files:
            logger.warning(f"未找到符合模式的检查点: {checkpoint_pattern}")
            return pd.DataFrame(), None, []
        
        logger.info(f"找到 {len(checkpoint_files)} 个检查点文件")

        # --- 3. 获取模型参数 ---
        num_labels = len(CONFIG.dataset_emotions(training_dataset_name))

        # --- 4. 循环评估每个检查点 ---
        results = []
        best_uar = -1.0
        best_conf_matrix = None
        best_checkpoint_name = ""

        for checkpoint_path in tqdm(checkpoint_files, desc="评估检查点"):
            epoch = checkpoint_path.stem.split('_')[-1]
            logger.info(f"\n{'='*60}")
            logger.info(f"评估检查点: {checkpoint_path.name}")
            logger.info(f"{'='*60}")
            
            try:
                # 创建模型并加载权重
                # 使用 ModelOps 而不是 cls，避免调用到 ContrastiveOps 的向后兼容方法
                model = ModelOps.create_or_load_model(
                    model_class=model_class,
                    num_labels=num_labels,
                    checkpoint_path=str(checkpoint_path)
                )
                
                # 创建训练器
                trainer = ModelOps.create_trainer(
                    trainer_class=trainer_class,
                    model=model,
                    alpha=alpha
                )
                
                # 在验证集上评估
                logger.info(f"在 {training_dataset_name} 验证集上评估...")
                val_uar, val_war, val_cm = trainer.eval(validation_loader)
                logger.info(f"验证集 - UAR: {val_uar*100:.2f}%, WAR: {val_war*100:.2f}%")
                
                # 在测试集上评估
                logger.info(f"在 {evaluation_dataset_name} 测试集上评估...")
                test_uar, test_war, test_cm = trainer.eval(test_loader)
                logger.info(f"测试集 - UAR: {test_uar*100:.2f}%, WAR: {test_war*100:.2f}%")
                
                # 记录结果
                results.append({
                    'checkpoint': checkpoint_path.name,
                    'epoch': epoch,
                    'val_uar': val_uar,
                    'val_war': val_war,
                    'test_uar': test_uar,
                    'test_war': test_war
                })
                
                # 更新最佳结果
                if test_uar > best_uar:
                    best_uar = test_uar
                    best_conf_matrix = test_cm
                    best_checkpoint_name = checkpoint_path.name
                    logger.info(f"发现新的最佳 UAR: {best_uar*100:.2f}% (来自 {checkpoint_path.name})")
                
                # 清理内存
                del model, trainer
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"评估 {checkpoint_path.name} 时出错: {e}", exc_info=True)
                continue

        # --- 5. 结果汇总 ---
        if not results:
            logger.warning("没有成功评估任何检查点")
            return pd.DataFrame(), None, []
        
        df_results = pd.DataFrame(results)
        logger.info("\n" + "="*80)
        logger.info("评估汇总:")
        logger.info("="*80)
        print(df_results.to_string(index=False))
        logger.info(f"\n最佳检查点: {best_checkpoint_name} (UAR: {best_uar*100:.2f}%)")
        
        return df_results, best_conf_matrix, test_emotions

