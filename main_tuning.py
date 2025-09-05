# main_tuning.py

import torch
import pandas as pd
import os
import gc

# 从您的模块中导入所需的工具
from core.config import CONFIG
from scripts.get_dataloaders import get_contrastive_dataloaders
from scripts.tuning import run_hyperparameter_trial # <-- 从您的工具箱导入函数

def main_tuning_loop():
    """
    执行完整的超参数alpha网格搜索的主函数。
    """
    # (这里的代码就是我上一条回复中提供给您的主循环代码)
    print("--- [阶段一：初始化] ---")
    CONFIG.load_config("config.yaml")
    
    alpha_values_to_test = [0.1, 0.5, 1.0, 2.0, 5.0]
    results_list = []
    results_filepath = os.path.join(CONFIG.project_root(), "alpha_tuning_results.csv")
    print(f"实验结果将实时保存在: {results_filepath}")

    print("\n--- [阶段二：预加载数据集] ---")
    dataloaders = get_contrastive_dataloaders(CONFIG.training_dataset_name())
    train_loader = dataloaders['train']
    validation_loader = dataloaders['validation']

    print("\n--- [阶段三：开始alpha网格搜索循环] ---")
    for alpha in alpha_values_to_test:
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            val_uar, val_war = run_hyperparameter_trial(
                alpha_value=alpha,
                train_loader=train_loader,
                validation_loader=validation_loader,
                config=CONFIG,
                num_epochs=CONFIG.training_epochs(),
                gradient_accumulation_steps=4
            )
            result_entry = {'alpha': alpha, 'validation_uar': val_uar, 'validation_war': val_war}
        except Exception as e:
            print(f"警告：alpha = {alpha} 的试验因错误而失败: {e}")
            result_entry = {'alpha': alpha, 'validation_uar': "Failed", 'validation_war': "Failed"}

        results_list.append(result_entry)
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(results_filepath, index=False)
        print(f"--- Alpha = {alpha} 的试验完成。结果已更新至CSV文件。 ---")

    print("\n--- [阶段四：所有试验完成] ---")
    print("最终的超参数搜索结果:")
    final_results = pd.read_csv(results_filepath)
    print(final_results)

# --- 脚本执行入口 ---
if __name__ == '__main__':
    main_tuning_loop()