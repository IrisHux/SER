import sys
import os
# 将项目根目录添加到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np
# import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from core.config import CONFIG

def load_data(model_type: str):
    """
    加载指定模型的嵌入向量和元数据。
    """
    print(f"--- 正在加载 [{model_type}] 数据 ---")
    
    # 1. 定义文件名
    embed_file = f"{model_type}_classification_embeddings.pt"
    meta_file = f"{model_type}_metadata.pkl"
    
    # 2. 检查文件是否存在
    if not os.path.exists(embed_file) or not os.path.exists(meta_file):
        print(f"[错误] 找不到文件 {embed_file} 或 {meta_file}")
        print(f"请先运行 'extract_features.py' 并设置 MODEL_TYPE='{model_type}'")
        return None, None

    # 3. 加载数据
    # .numpy() 是必须的，t-SNE 在 NumPy 数组上运行
    X = torch.load(embed_file).cpu().numpy()
    df_meta = pd.read_pickle(meta_file)
    
    print(f"加载成功: {len(X)} 个特征向量 (维度: {X.shape[1]})")
    
    return X, df_meta

def run_tsne(X: np.ndarray) -> np.ndarray:
    """
    对高维特征矩阵 X 运行 t-SNE。
    这是一个计算密集型过程。
    """
    print(f"--- 正在对 {len(X)} 个样本运行 t-SNE... ---")
    
    tsne = TSNE(
        n_components=2,     # 我们想要一个 2D 图像
        perplexity=30,      # 一个标准的默认值，您可以尝试调整 (例如 15-50)
        max_iter=1000,        # 迭代次数
        random_state=42,    # 关键！确保每次运行结果都一样
        n_jobs=-1,           # 使用所有可用的 CPU 核心
        verbose=1
    )
    
    # # tqdm 用于显示进度条
    # with tqdm(total=1) as pbar:
    #     def update_pbar(iter):
    #         pbar.set_description(f"t-SNE 迭代 {iter}/{tsne.max_iter}")
    #         pbar.update(iter - pbar.n)
        
    #     # t-SNE 没有原生的进度条，这是一个小技巧
    #     # 注意：fit_transform 是唯一会显示进度的
    #     tsne.verbose = update_pbar
    X_2d = tsne.fit_transform(X)
    # pbar.set_description("t-SNE 计算完成")
    print("--- t-SNE 计算完成 ---")

    return X_2d

def plot_tsne_visualizations(X_2d: np.ndarray, df_meta: pd.DataFrame, model_type: str):
    """
    使用 seaborn 绘制并保存 t-SNE 结果。
    """
    print(f"--- 正在为 [{model_type}] 生成 t-SNE 图像 ---")
    
    # 1. 将 2D 坐标添加回 DataFrame 以便绘图
    df_plot = df_meta.copy()
    df_plot['tsne-1'] = X_2d[:, 0]
    df_plot['tsne-2'] = X_2d[:, 1]
    
    # 2. 获取保存路径
    save_dir = CONFIG.save_plots_location()
    
    # --- 图 1: 按情感着色 (核心视图) ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='tsne-1',
        y='tsne-2',
        hue='emotion',        # 按 'emotion' 列着色
        style='dataset_source', # 用不同形状区分 IEMOCAP 和 CREMA-D
        data=df_plot,
        alpha=0.7,
        s=50
    )
    plt.title(f"t-SNE of [{model_type}] Features (Colored by Emotion)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    save_path_emotion = os.path.join(save_dir, f"tsne_{model_type}_by_emotion.png")
    plt.savefig(save_path_emotion)
    plt.close()
    print(f"图像已保存到: {save_path_emotion}")

    # --- 图 2: 按数据集着色 (用于检查 domain bias) ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='tsne-1',
        y='tsne-2',
        hue='dataset_source', # 按 'dataset_source' 列着色
        style='emotion',      # 用不同形状区分情感
        data=df_plot,
        alpha=0.7,
        s=50
    )
    plt.title(f"t-SNE of [{model_type}] Features (Colored by Dataset)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    save_path_dataset = os.path.join(save_dir, f"tsne_{model_type}_by_dataset.png")
    plt.savefig(save_path_dataset)
    plt.close()
    print(f"图像已保存到: {save_path_dataset}")

def main():
    # 加载全局配置以获取保存路径
    try:
        CONFIG.load_config("config.yaml")
    except FileNotFoundError:
        print("[错误] 找不到 config.yaml。请在项目根目录运行此脚本。")
        return

    model_types_to_plot = ["baseline", "lgca"]
    
    for model_type in model_types_to_plot:
        X, df_meta = load_data(model_type)
        if X is None:
            continue
        
        X_2d = run_tsne(X)
        plot_tsne_visualizations(X_2d, df_meta, model_type)
        
    print("\n--- t-SNE 可视化全部完成！---")

if __name__ == "__main__":
    main()