# main_contrastive.py

import torch
import gc
import os
import logging
from tqdm.contrib.logging import _TqdmLoggingHandler
import warnings # <-- 新增

warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 添加 tqdm 的日志处理器，以确保日志不会破坏进度条
logging.getLogger().addHandler(_TqdmLoggingHandler())


# 导入所有必要的模块
from core.config import CONFIG, device
from contrastive.model import ContrastiveModel
from contrastive.trainer import ContrastiveTrainer
# 确保您已经创建了这个新的数据加载器脚本
from scripts.get_dataloaders import get_contrastive_dataloaders

def run_experiment():
    """
    执行完整的LGCA框架训练和评估实验。
    """
    # --- 1. 加载配置并设置环境 ---
    try:
        CONFIG.load_config("config.yaml")
        logging.info("配置文件 'config.yaml' 加载成功。")
    except FileNotFoundError:
        logging.error("错误：找不到 'config.yaml' 文件。请确保该文件存在于项目根目录。")
        return

    # 设置内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()
    gc.collect()

    # --- 2. 加载数据集 ---
    try:
        training_dataset_name = CONFIG.training_dataset_name()
        logging.info(f"--- 正在为训练集 '{training_dataset_name}' 准备Dataloaders ---")
        iemocap_loaders = get_contrastive_dataloaders(training_dataset_name)
        train_loader = iemocap_loaders['train']
        validation_loader = iemocap_loaders['validation']

        evaluation_dataset_name = CONFIG.evaluation_dataset_name()
        logging.info(f"--- 正在为评估集 '{evaluation_dataset_name}' 准备Dataloaders ---")
        cremad_loaders = get_contrastive_dataloaders(evaluation_dataset_name)
        evaluation_loader = cremad_loaders['evaluation']
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        return

    # --- 3. 实例化模型和训练器 ---
    logging.info("--- 正在初始化 ContrastiveModel 和 ContrastiveTrainer ---")
    try:
        num_labels = len(CONFIG.dataset_emotions(training_dataset_name))
        model = ContrastiveModel(num_labels=num_labels).to(device)
        
        # # 启用梯度检查点以节省显存
        # model.audio_encoder.gradient_checkpointing_enable()
        # model.text_encoder.gradient_checkpointing_enable()
        # logging.info("已为声学和文本编码器启用梯度检查点。")

        trainer = ContrastiveTrainer(
            model=model,
            num_epochs=CONFIG.training_epochs(),
            learning_rate=CONFIG.learning_rate(),
            optimizer_type=CONFIG.optimizer_type(),
            # 配合config.yaml中的batch_size来设置，例如batch_size=2, steps=4 -> 有效批次=8
            gradient_accumulation_steps=4 
        )
    except Exception as e:
        logging.error(f"模型或训练器实例化失败: {e}")
        return

    # --- 4. 开始训练 ---
    logging.info(f"--- 开始在 '{training_dataset_name}' 上进行训练，共 {CONFIG.training_epochs()} 个 Epochs ---")
    # trainer.train(train_loader)
        # *** 核心修改点：将 validation_loader 传递给 train 方法 ***
    trainer.train(train_loader, validation_loader) 
    
    logging.info("--- 训练完成 ---")

    # --- 5. 保存模型 ---
    model_save_path = os.path.join(CONFIG.saved_models_location(), "lgca_model_final.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"模型已保存至: {model_save_path}")

    # --- 6. 进行评估 ---
    iemocap_emotions = CONFIG.dataset_emotions(training_dataset_name)
    cremad_emotions = CONFIG.dataset_emotions(evaluation_dataset_name)

    logging.info(f"--- 在 '{training_dataset_name}' 验证集上进行评估 ---")
    trainer.eval(validation_loader, labels=iemocap_emotions)

    logging.info(f"--- 在 '{evaluation_dataset_name}' 测试集上进行零样本评估 ---")
    trainer.eval(evaluation_loader, labels=cremad_emotions)

    logging.info("--- 实验流程全部完成！ ---")


if __name__ == '__main__':
    run_experiment()