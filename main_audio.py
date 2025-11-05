

import os
import torch
import gc

from core.config import CONFIG, device
from scripts.get_dataloaders import get_dataloaders
from scripts.preprocess_data import process_raw_data_to_pickle
from audio.baseline_model import AudioBaselineModel
from audio.trainer import MemoryOptimizedAudioBaselineTrainer

import warnings

# 屏蔽所有 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    """
    主执行函数，包含模型训练和评估的全部流程。
    """
    try:
        # --- 步骤 0: 环境设置与配置加载 ---
        print("--- [步骤 0] 环境设置与配置加载 ---")
        # 设置CUDA内存优化 (可选，但推荐)
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        gc.collect()

        # 加载 config.yaml 配置文件
        CONFIG.load_config("config.yaml")
        print(f"base_size := {CONFIG.dataloader_dict()['batch_size']}")

        # --- 步骤 1: 确保原始数据信息文件存在 (一次性运行) ---
        print("\n--- [步骤 1] 检查并生成原始数据信息文件 ---")
        training_dataset_name = CONFIG.training_dataset_name()
        evaluation_dataset_name = CONFIG.evaluation_dataset_name()

        # 为训练集生成 _raw.pkl
        train_raw_filename = f"{training_dataset_name.split('_')[0].lower()}_raw.pkl"
        train_raw_filepath = os.path.join(CONFIG.dataset_preprocessed_dir_path(training_dataset_name), train_raw_filename)
        if not os.path.exists(train_raw_filepath):
            print(f"未找到 {train_raw_filepath}，正在生成...")
            process_raw_data_to_pickle(training_dataset_name, train_raw_filename)
        else:
            print(f"已找到 {train_raw_filepath}，跳过生成。")

        # 为评估集生成 _raw.pkl
        eval_raw_filename = f"{evaluation_dataset_name.split('_')[0].lower()}_raw.pkl"
        eval_raw_filepath = os.path.join(CONFIG.dataset_preprocessed_dir_path(evaluation_dataset_name), eval_raw_filename)
        if not os.path.exists(eval_raw_filepath):
            print(f"未找到 {eval_raw_filepath}，正在生成...")
            process_raw_data_to_pickle(evaluation_dataset_name, eval_raw_filename)
        else:
            print(f"已找到 {eval_raw_filepath}，跳过生成。")

        # --- 步骤 2: 加载数据集 (采用实时处理) ---
        print(f"\n--- [步骤 2] 加载数据集 (实时处理模式) ---")
        print(f"\n--- 正在加载 '{training_dataset_name}' 数据集 ---")
        iemocap_loaders = get_dataloaders(training_dataset_name)
        train_loader = iemocap_loaders['train']
        validation_loader = iemocap_loaders['validation']

        print(f"\n--- 正在加载 '{evaluation_dataset_name}' 数据集 ---")
        cremad_loaders = get_dataloaders(evaluation_dataset_name)
        evaluation_loader = cremad_loaders['evaluation']

        # --- 步骤 3: 实例化模型和训练器 ---
        print("\n--- [步骤 3] 初始化基线模型和训练器 ---")
        
        torch.cuda.empty_cache()
        gc.collect()

        iemocap_emotions = CONFIG.dataset_emotions(training_dataset_name)
        num_labels = len(iemocap_emotions)
        baseline_model = AudioBaselineModel(num_labels=num_labels).to(device)

        baseline_trainer = MemoryOptimizedAudioBaselineTrainer(
            model=baseline_model,
            num_epochs=10,
            learning_rate=1e-4,
            optimizer_type="Adam",
            gradient_accumulation_steps=4
        )

        # --- 步骤 4: 训练模型 ---
        print("\n--- [步骤 4] 开始在 IEMOCAP 上训练基线模型 ---")
        training_history = baseline_trainer.train(train_loader, validation_loader)

        # --- 步骤 5: 保存训练好的模型 ---
        print("\n--- [步骤 5] 保存训练好的模型 ---")
        model_save_dir = CONFIG.saved_models_location()
        model_save_path = os.path.join(model_save_dir, "Audio_Baseline_trained_model.pt")
        
        # 保存模型的state_dict
        torch.save({
            'model_state_dict': baseline_model.state_dict(),
            'optimizer_state_dict': baseline_trainer._optimizer.state_dict(),
            'training_history': training_history,
            'num_labels': num_labels,
            'emotions': iemocap_emotions
        }, model_save_path)
        print(f"模型已保存到: {model_save_path}")

        # --- 步骤 6: 在 IEMOCAP 验证集上评估 ---
        print("\n--- [步骤 6] 在 IEMOCAP 验证集上评估模型性能 ---")
        iemocap_uar, iemocap_war, _ = baseline_trainer.eval(validation_loader, labels=iemocap_emotions)

        # --- 步骤 7: 清理内存并加载保存的模型用于CREMA-D评估 ---
        print("\n--- [步骤 7] 清理内存并准备在 CREMA-D 上评估 ---")
        # 删除当前模型和训练器以释放内存
        del baseline_model
        del baseline_trainer
        torch.cuda.empty_cache()
        gc.collect()

        # --- 步骤 8: 加载训练好的模型 ---
        print("\n--- [步骤 8] 加载训练好的模型 ---")
        checkpoint = torch.load(model_save_path, map_location=device)
        loaded_model = AudioBaselineModel(num_labels=checkpoint['num_labels']).to(device)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {model_save_path} 加载")

        # --- 步骤 9: 在 CREMA-D 测试集上进行零样本评估 ---
        print("\n--- [步骤 9] 在 CREMA-D 测试集上进行零样本评估 ---")
        cremad_emotions = CONFIG.dataset_emotions(evaluation_dataset_name)
        
        # 创建一个用于评估的trainer（不需要训练，只需要eval方法）
        eval_trainer = MemoryOptimizedAudioBaselineTrainer(
            model=loaded_model,
            num_epochs=0,  # 不进行训练
            learning_rate=1e-4,
            optimizer_type="Adam",
            gradient_accumulation_steps=4
        )
        
        print("开始在 CREMA-D 上评估...")
        cremad_uar, cremad_war, _ = eval_trainer.eval(evaluation_loader, labels=cremad_emotions)
        
        # --- 步骤 10: 打印评估总结 ---
        print("\n" + "="*60)
        print("评估结果总结:")
        print("="*60)
        print(f"CREMA-D 数据集:")
        print(f"  - UAR (Unweighted Average Recall): {cremad_uar:.4f}")
        print(f"  - WAR (Weighted Average Recall/Accuracy): {cremad_war:.4f}")
        print("="*60)

        print("\n--- 基线模型训练和评估完成！ ---")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n[ERROR] CUDA内存不足: {e}")
        print("建议:")
        print("1. 在 config.yaml 中进一步减小 batch_size (例如到 1 或 2)。")
        print("2. 在 audio/trainer.py 的 MemoryOptimizedAudioBaselineTrainer 中增加 gradient_accumulation_steps。")
        print("3. 在 config.yaml 中减小 learning_rate (例如 1e-5)。")
        print("4. 如果仍然失败，可以在 AudioBaselineTrainer 中重新冻结特征提取器 (`self.model.wavlm.freeze_feature_extractor()`) 作为最后的手段。")

    except Exception as e:
        print(f"\n[ERROR] 训练过程中出现未知错误: {e}")
        raise e

    finally:
        # 清理内存
        print("\n--- 执行最终内存清理 ---")
        torch.cuda.empty_cache()
        gc.collect()


# ==============================================================================
# 脚本的入口点
# ==============================================================================
if __name__ == '__main__':
    # 当这个.py文件被直接运行时，以下代码块才会被执行
    main()