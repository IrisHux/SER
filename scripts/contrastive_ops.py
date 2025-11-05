# contrastive_ops.py

import torch
import gc
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入您项目中的核心模块
from core.config import CONFIG, device
from contrastive.model import MemoryOptimizedContrastiveModel
from contrastive.trainer import ContrastiveTrainer
from scripts.get_dataloaders import get_contrastive_dataloaders

logger = logging.getLogger(__name__)

class ContrastiveOps:
    """
    封装所有与 ContrastiveModel 相关的创建、训练和评估操作。
    """

    @classmethod
    def create_or_load_model(cls, load_path: str = None) -> MemoryOptimizedContrastiveModel:
        """
        创建新模型或从 state_dict 加载模型。
        
        Args:
            load_path (str, optional):
                要加载的检查点文件名 (例如 'contrastive_lgca_epoch_5.pt')。
                它将在 CONFIG.saved_ckpt_location() 目录中查找。
        """
        # 总是需要 num_labels 来初始化模型结构
        training_dataset_name = CONFIG.training_dataset_name()
        num_labels = len(CONFIG.dataset_emotions(training_dataset_name))
        
        model = MemoryOptimizedContrastiveModel(num_labels=num_labels).to(device)
        
        if load_path:
            full_path = os.path.join(CONFIG.saved_ckpt_location(), load_path)
            if not os.path.exists(full_path):
                logger.error(f"错误：找不到检查点文件 {full_path}")
                raise FileNotFoundError(f"找不到检查点文件 {full_path}")
                
            logger.info(f"--- 正在从 {full_path} 加载模型权重 ---")
            model.load_state_dict(torch.load(full_path, map_location=device))
        else:
            logger.info("--- 正在创建新的 MemoryOptimizedContrastiveModel ---")
            
        return model

    @classmethod
    def create_trainer(cls, model: MemoryOptimizedContrastiveModel) -> ContrastiveTrainer:
        """
        根据 CONFIG 配置创建一个 ContrastiveTrainer 实例。
        
        Args:
            model (MemoryOptimizedContrastiveModel):
                要训练或评估的模型实例。
        """
        logger.info("--- 正在初始化 ContrastiveTrainer ---")
        trainer = ContrastiveTrainer(
            model=model,
            num_epochs=CONFIG.training_epochs(),
            learning_rate=CONFIG.learning_rate(),
            alpha=CONFIG.llgca_loss_alpha(),
            optimizer_type=CONFIG.optimizer_type(),
            gradient_accumulation_steps=2 # 您可以将其也移至 config.yaml
        )
        return trainer

    @classmethod
    def train(cls, trainer: ContrastiveTrainer):
        """
        使用 'train' 和 'validation' 数据集运行训练过程。
        """
        training_dataset_name = CONFIG.training_dataset_name()
        logger.info(f"--- 正在为训练集 '{training_dataset_name}' 准备 Dataloaders ---")
        try:
            iemocap_loaders = get_contrastive_dataloaders(
                training_dataset_name, 
                use_audio_augmentation=True #是否使用数据增强
            )
            train_loader = iemocap_loaders['train']
            validation_loader = iemocap_loaders['validation']
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return

        logger.info(f"--- 开始在 '{training_dataset_name}' 上进行训练，共 {trainer._num_epochs} 个 Epochs ---")
        # 假设 trainer.train 会处理训练、验证循环和模型保存
        trainer.train(train_loader, validation_loader) 
        logger.info("--- 训练完成 ---")

    @classmethod
    def evaluate(cls, trainer: ContrastiveTrainer, dataset_split: str):
        """
        在指定的数据集拆分上评估模型 (例如 'validation' 或 'evaluation')。

        Args:
            trainer (ContrastiveTrainer): 包含要评估的模型的训练器。
            dataset_split (str): 'validation' 或 'evaluation'。
            
        Returns:
            tuple: (uar, war)
        """
        if dataset_split == 'validation':
            dataset_name = CONFIG.training_dataset_name()
        elif dataset_split == 'evaluation':
            dataset_name = CONFIG.evaluation_dataset_name()
        else:
            logger.error(f"未知的 dataset_split: {dataset_split}")
            return None, None

        logger.info(f"--- 正在为 '{dataset_name}' 的 '{dataset_split}' 拆分准备 Dataloader ---")
        try:
            loaders = get_contrastive_dataloaders(dataset_name)
            loader = loaders[dataset_split]
            emotions = CONFIG.dataset_emotions(dataset_name)
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return None, None

        logger.info(f"--- 正在 '{dataset_split}' 拆分上评估模型 ---")
        # eval 返回 (uar, war, conf_matrix)
        uar, war, conf_matrix = trainer.eval(loader, labels=emotions)
        logger.info(f"评估结果 ({dataset_split}) - UAR: {uar:.4f}, WAR: {war:.4f}")
        return uar, war

    @classmethod
    def evaluate_all_checkpoints(cls, evaluation_dataset_name: str) -> pd.DataFrame:
        """
        加载 'checkpoints/' 目录下的所有模型，并在测试集 (evaluation) 上进行评估。
        
        Args:
            evaluation_dataset_name (str): 在 config 中定义的评估数据集的名称。
        
        Returns:
            pd.DataFrame: 包含 'checkpoint', 'test_uar', 'test_war' 的数据框。
        """
        logger.info("\n--- [阶段：开始在测试集上评估所有检查点] ---")
        
        # --- 1. 加载测试数据集 ---
        logger.info(f"--- 正在为测试集 '{evaluation_dataset_name}' 准备Dataloader ---")
        try:
            cremad_loaders = get_contrastive_dataloaders(evaluation_dataset_name)
            evaluation_loader = cremad_loaders['evaluation']
            cremad_emotions = CONFIG.dataset_emotions(evaluation_dataset_name)
            logger.info(f"测试集 '{evaluation_dataset_name}' 加载完毕。")
        except Exception as e:
            logger.error(f"测试数据加载失败: {e}")
            return pd.DataFrame()

        # --- 2. 查找所有模型检查点 ---
        checkpoint_dir = CONFIG.saved_ckpt_location()
        if not os.path.exists(checkpoint_dir):
            logger.error(f"错误：找不到检查点目录 '{checkpoint_dir}'。")
            return pd.DataFrame()
            
        checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                            if f.lower().startswith('contrastive_lgca') and f.lower().endswith('.pt')]
        try:
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        except ValueError:
            logger.warning("部分检查点文件名不规范，将按默认字母顺序排序。")

        if not checkpoint_files:
            logger.error(f"在 '{checkpoint_dir}' 目录中没有找到任何检查点文件 (.pt)。")
            return pd.DataFrame()
        
        logger.info(f"找到了 {len(checkpoint_files)} 个检查点文件，将逐一进行评估。")

        # --- 3. 循环评估每个检查点 ---
        results_list = []

        # ⭐️ 新增：用于跟踪最佳UAR和对应的混淆矩阵
        best_uar = -1.0
        best_conf_matrix = None
        best_cm_checkpoint_name = ""

        for checkpoint_file in tqdm(checkpoint_files, desc="评估所有检查点"):
            
            # 1. 实例化一个全新的模型并加载权重
            # (我们使用 load_path 参数)
            model = cls.create_or_load_model(load_path=checkpoint_file)
            
            # 2. 实例化一个Trainer用于评估
            # (训练参数如lr, epochs在评估时无关紧要)
            trainer = cls.create_trainer(model)
            
            # 3. 调用 eval 方法进行评估
            logger.info(f"\n--- 正在评估: {checkpoint_file} ---")
            test_uar, test_war, conf_matrix = trainer.eval(evaluation_loader, labels=cremad_emotions)
            
            # 4. 记录结果
            results_list.append({
                'checkpoint': checkpoint_file,
                'test_uar': test_uar,
                'test_war': test_war
            })

            # 5. ⭐️ 新增：检查并保存最佳的混淆矩阵
            if test_uar > best_uar:
                best_uar = test_uar
                best_conf_matrix = conf_matrix
                best_cm_checkpoint_name = checkpoint_file
                logger.info(f"--- 发现新的最佳 UAR: {best_uar:.4f} (来自 {checkpoint_file}) ---")

            # 6. 清理内存
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

        # --- 4. 结果汇总 ---
        logger.info("\n--- [阶段：所有检查点评估完成，汇总结果] ---")
        if not results_list:
            logger.warning("没有评估任何模型，无法生成报告。")
            return pd.DataFrame()

        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values(by='test_uar', ascending=False).reset_index(drop=True)
        # 7. ⭐️ 修改：返回df、最佳矩阵和标签
        logger.info(f"--- 最佳混淆矩阵来自: {best_cm_checkpoint_name} ---")
        
        return results_df, best_conf_matrix, cremad_emotions