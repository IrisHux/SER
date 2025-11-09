# run_contrastive.py

import torch
import gc
import os
import logging
import numpy as np
import random
import warnings

# 导入您项目中的核心模块
from core.config import CONFIG, device
from vizualisers.plots import PlotVisualizer
from contrastive.model import setup_memory_optimization, MemoryOptimizedContrastiveModel
from contrastive.trainer import ContrastiveTrainer
from scripts.contrastive_ops import ModelOps  # 使用新的通用类
from scripts.get_dataloaders import get_contrastive_dataloaders

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_env():
    """
    加载配置、设置随机种子并准备环境。
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # --- 1. 加载配置 ---
    try:
        CONFIG.load_config("config.yaml")
        logger.info("配置文件 'config.yaml' 加载成功。")
    except FileNotFoundError:
        logger.error("错误：找不到 'config.yaml' 文件。请确保该文件存在于项目根目录。")
        raise
        
    # --- 2. 设置随机种子 ---
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # --- 3. 设置内存和GPU ---
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    setup_memory_optimization()
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("环境、配置和随机种子准备就绪。")


if __name__ == "__main__":
    
    # 0. 准备环境
    prepare_env()

    # --- 您可以在此处控制要运行的阶段 ---
    RUN_TRAINING = True
    RUN_EVALUATE_ALL_CHECKPOINTS = True
    # ------------------------------------

    trainer_for_validation = None

    # === 阶段 1: 训练模型 ===
    # (对应 main_contrastive.py)
    if RUN_TRAINING:
        logger.info("\n==================== [阶段 1: 开始训练] ====================")
        
        # 获取配置参数
        training_dataset_name = CONFIG.training_dataset_name()
        num_labels = len(CONFIG.dataset_emotions(training_dataset_name))
        
        # 1.1. 创建一个新模型（使用 ModelOps）
        model = ModelOps.create_or_load_model(
            model_class=MemoryOptimizedContrastiveModel,
            num_labels=num_labels
        )
        
        # 1.2. 为模型创建训练器（使用 ModelOps）
        trainer = ModelOps.create_trainer(
            trainer_class=ContrastiveTrainer,
            model=model
        )
        
        # 1.3. 运行训练（使用 ModelOps）
        ModelOps.train(
            trainer=trainer,
            training_dataset_name=training_dataset_name,
            dataloader_func=get_contrastive_dataloaders
        )
        
        # 1.4. 训练完成后，立即在验证集上运行一次最终评估
        logger.info("--- 对训练完成的最终模型在验证集上进行评估 ---")
        ModelOps.evaluate(
            trainer=trainer,
            dataset_split='validation',
            dataset_name=training_dataset_name,
            dataloader_func=get_contrastive_dataloaders
        )
        logger.info("==================== [阶段 1: 训练完成] ====================\n")


    # === 阶段 2: 评估所有检查点 ===
    # (对应 evaluate_checkpoints.py)
    if RUN_EVALUATE_ALL_CHECKPOINTS:
        logger.info("\n==================== [阶段 2: 评估所有检查点] ====================")
        
        # 获取配置参数
        training_dataset_name = CONFIG.training_dataset_name()
        evaluation_dataset_name = CONFIG.evaluation_dataset_name()
        
        # 2.1. 运行批量评估（使用 ModelOps）
        results_df, best_cm, eval_labels = ModelOps.evaluate_all_checkpoints(
            model_class=MemoryOptimizedContrastiveModel,
            trainer_class=ContrastiveTrainer,
            checkpoint_pattern='Contrastive_LGCA_model_epoch_*.pt',
            training_dataset_name=training_dataset_name,
            evaluation_dataset_name=evaluation_dataset_name,
            dataloader_func=get_contrastive_dataloaders
        )
        
        # 2.2. 保存和打印结果
        if results_df is not None and not results_df.empty:
            save_path = os.path.join(CONFIG.save_tables_location(), "final_test_evaluation_results.csv")
            results_df.to_csv(save_path, index=False)
            
            print("\n==================== 最终测试集评估结果汇总 ====================")
            print(results_df)
            print(f"\n评估结果已保存至: {save_path}")
            
            # 找到并高亮显示最佳模型
            best_model_stats = results_df.iloc[0]
            best_checkpoint_name = best_model_stats['checkpoint']

            # 3. ⭐️ 保存最佳模型的混淆矩阵
            if best_cm is not None:
                plot_filename = f"best_model_cm_{best_checkpoint_name.replace('.pt', '.png')}"
                best_plot_save_path = os.path.join(CONFIG.save_plots_location(), plot_filename)
                
                logger.info(f"\n--- 正在为最佳模型 '{best_checkpoint_name}' 保存混淆矩阵 ---")
                
                try:
                    # 使用 PlotVisualizer 绘制混淆矩阵（会自动保存到 pictures/ 目录）
                    PlotVisualizer.plot_confusion_matrix(
                        confusion_matrix=best_cm,
                        labels=eval_labels,
                        filename=plot_filename
                    )
                    
                    logger.info(f"✅ 最佳模型的混淆矩阵已保存至: {best_plot_save_path}")
                
                except Exception as e:
                    logger.error(f"为最佳模型保存混淆矩阵时出错: {e}")

            print("\n==================== 最佳模型表现 ====================")
            print(f"🏆 最佳模型检查点: {best_checkpoint_name}")
            print(f"   - 最佳测试集 UAR: {best_model_stats['test_uar']:.4f}")
            print(f"   - 对应的测试集 WAR: {best_model_stats['test_war']:.4f}")
            print("==========================================================")
            
            # 4. ⭐️ 新增：保存最佳模型检查点到 saved_models_location
            try:
                # 源文件路径（在checkpoints目录中）
                source_checkpoint_path = os.path.join(CONFIG.saved_ckpt_location(), best_checkpoint_name)
                # 目标文件路径（在saved_models目录中，使用新名称）
                target_model_path = os.path.join(CONFIG.saved_models_location(), "Contrastive_LGCA_model.pt")
                
                logger.info(f"\n--- 正在保存最佳模型到: {target_model_path} ---")
                
                # 加载并保存模型（这样可以确保文件完整性）
                best_model_state = torch.load(source_checkpoint_path, map_location=device)
                torch.save(best_model_state, target_model_path)
                
                logger.info(f"✅ 最佳模型已成功保存为: Contrastive_LGCA_model.pt")
                print(f"\n✅ 最佳模型已保存至: {target_model_path}")
                
            except Exception as e:
                logger.error(f"保存最佳模型时出错: {e}")
        else:
            logger.warning("未生成评估结果。")
        logger.info("==================== [阶段 2: 评估完成] ====================")