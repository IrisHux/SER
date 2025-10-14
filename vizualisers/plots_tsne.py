# vizualisers/plot_tsne.py

import os
from matplotlib.lines import Line2D
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from core.config import CONFIG

def run_tsne(embeddings, perplexity, n_iter, random_state):
    """在嵌入向量上运行 t-SNE 降维"""
    print(f"--- 开始计算 t-SNE (Perplexity={perplexity}, Iterations={n_iter}) ---")
    print(f"输入数据维度: {embeddings.shape}")
    
    # 确保数据是 float32 的 numpy 数组
    embeddings_np = embeddings.astype(np.float32)
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,  # sklearn newer versions use `max_iter` instead of `n_iter`
        init='pca', # PCA 初始化更稳定
        random_state=random_state,
        verbose=1
    )
    
    tsne_results = tsne.fit_transform(embeddings_np)
    print("--- t-SNE 计算完成 ---")
    return tsne_results

def plot_tsne_charts(df, perplexity):
    """使用包含 t-SNE 坐标的 DataFrame 绘制三个图表"""
    
    print("--- 开始生成图表 ---")
    
    # 设置绘图风格
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # 创建一个 2x2 的子图画布
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(f"Model acoustic embeddings t-SNE visualization (Perplexity={perplexity})", fontsize=20, y=1.05)

    # --- 图1: 情感判别性 (CREMA-D) ---
    ax1 = axes[0, 0]
    crema_df = df[df['dataset_source'] == 'CREMA-D']
    sns.scatterplot(
        data=crema_df,
        x='tsne-1',
        y='tsne-2',
        hue='emotion',
        ax=ax1,
        palette='deep',
        s=20,          # 减小点的大小
        alpha=0.7
    )
    ax1.set_title("Plot 1: Emotion separability (CREMA-D)", fontsize=16)
    ax1.legend(loc='best', markerscale=2)

    # --- 图2: 领域不变性 (IEMOCAP vs CREMA-D) ---
    ax2 = axes[0,1]
    sns.scatterplot(
        data=df,
        x='tsne-1',
        y='tsne-2',
        hue='dataset_source',
        ax=ax2,
        palette='bright',
        s=20,
        alpha=0.7
    )
    ax2.set_title("Plot 2: Domain invariance (IEMOCAP vs CREMA-D)", fontsize=16)
    ax2.legend(loc='best', markerscale=2)

    # --- 图4: 情感判别性 (IEMOCAP) ---
    ax4 = axes[1,0]
    iemocap_df = df[df['dataset_source'] == 'IEMOCAP']
    sns.scatterplot(
        data=iemocap_df,
        x='tsne-1',
        y='tsne-2',
        hue='emotion',
        ax=ax4,
        palette='deep',
        s=20,          # 减小点的大小
        alpha=0.7
    )
    ax4.set_title("Plot 3: Emotion separability (IEMOCAP)", fontsize=16)
    ax4.legend(loc='best', markerscale=2)

    # --- 图3: 石蕊测试 (CREMA-D "一文多情" 样本) ---
    ax3 = axes[1, 1]
    # 筛选逻辑
    crema_df_text = df[df['dataset_source'] == 'CREMA-D'][['text', 'emotion', 'tsne-1', 'tsne-2']]
    text_emotion_counts = crema_df_text.groupby('text')['emotion'].nunique()
    litmus_texts = text_emotion_counts[text_emotion_counts > 1].index
    litmus_df = crema_df_text[crema_df_text['text'].isin(litmus_texts)]
    
    if litmus_df.empty:
        print("[警告] 未能在 CREMA-D 中找到 '一文多情' 的样本，图3将为空。")
        ax3.text(0.5, 0.5, "未找到 '一文多情' 样本", 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=ax3.transAxes, color='red')
    else:
         sns.scatterplot(
            data=litmus_df,
            x='tsne-1',
            y='tsne-2',
            hue='emotion',
            style='text', # 用形状区分不同句子 (如果句子不多的话)
            ax=ax3,
            palette='deep',
            s=80,         # 放大点的大小以便观察
            alpha=0.9
        )
         # 由于文本可能很多，图例可能会爆炸，这里选择不显示 style 的图例
         ax3.legend(title='Emotion', loc='center left', bbox_to_anchor=(1.02, 0.5), markerscale=2)
         
    ax3.set_title(f"Plot 4: Litmus test (CREMA-D 'same text, different emotion' samples, {len(litmus_texts)} sentences)", fontsize=16)

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")


    plt.tight_layout(rect=[0, 0, 1, 0.98], w_pad=3.0, h_pad=3.0)  # 留出顶部空间给 suptitle
    plt.savefig(os.path.join(CONFIG.save_plots_location(), "tsne_visualization.png"), dpi=300, bbox_inches='tight')
    print(f"--- 图表已保存为 tsne_visualization.png ---")
    plt.show()



