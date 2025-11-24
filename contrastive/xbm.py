# contrastive/xbm.py
"""
Cross-Batch Memory (XBM) 实现

论文参考: Cross-Batch Memory for Embedding Learning (CVPR 2020)
https://arxiv.org/abs/1912.06798

XBM通过维护一个固定大小的记忆库，存储历史批次的特征和标签，
从而扩大对比学习的负样本池，提升训练效果。
"""

import torch
import torch.nn.functional as F
from typing import Optional


class XBM:
    """
    Cross-Batch Memory (跨批次记忆库)
    
    数据结构:
    - Feature Bank (M_F): 存储样本的特征向量，维度 (K, D)
    - Label Bank (M_L): 存储对应样本的情感标签，维度 (K,)
    
    其中 K 是记忆库容量，D 是特征维度。
    """
    
    def __init__(self, memory_size: int, feat_dim: int, device: torch.device):
        """
        初始化XBM记忆库。
        
        Args:
            memory_size (int): 记忆库容量 K（建议16384或更大）
            feat_dim (int): 特征维度 D（例如投影头输出的128或256）
            device (torch.device): 存储设备（通常是cuda）
        """
        self.K = memory_size
        self.D = feat_dim
        self.device = device
        
        # 初始化特征队列和标签队列
        self.feats = torch.zeros(self.K, self.D, dtype=torch.float32, device=device)
        self.labels = torch.zeros(self.K, dtype=torch.long, device=device)
        
        # 队列指针（指向下一个要被替换的位置）
        self.ptr = 0
        
        # 记忆库是否已满的标志
        self.is_full = False
        
        # 计算显存占用
        vram_mb = (self.K * self.D * 4) / (1024 ** 2)  # FP32 占 4 字节
        print(f"[XBM] 初始化记忆库: K={self.K}, D={self.D}")
        print(f"[XBM] 预计显存占用: {vram_mb:.2f} MB")
    
    @torch.no_grad()
    def enqueue_dequeue(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        更新记忆库：入队新特征，出队旧特征（FIFO策略）。
        
        关键设计:
        1. 必须在调用前对 feats 执行 .detach() 操作，避免梯度回传
        2. 支持批量更新（一次性入队整个mini-batch）
        
        Args:
            feats (torch.Tensor): 当前批次的特征，形状 (B, D)，已经过L2归一化
            labels (torch.Tensor): 当前批次的标签，形状 (B,)
        """
        batch_size = feats.size(0)
        
        # 确保特征已从计算图中分离
        assert not feats.requires_grad, "特征必须先detach()再入队！"
        
        # 计算要更新的索引范围
        end_ptr = self.ptr + batch_size
        
        if end_ptr <= self.K:
            # 情况1: 不需要环绕
            self.feats[self.ptr:end_ptr] = feats
            self.labels[self.ptr:end_ptr] = labels
            self.ptr = end_ptr % self.K  # 自动环绕
        else:
            # 情况2: 需要环绕到队列开头
            overflow = end_ptr - self.K
            self.feats[self.ptr:self.K] = feats[:self.K - self.ptr]
            self.labels[self.ptr:self.K] = labels[:self.K - self.ptr]
            self.feats[0:overflow] = feats[self.K - self.ptr:]
            self.labels[0:overflow] = labels[self.K - self.ptr:]
            self.ptr = overflow
        
        # 标记记忆库已满（至少经历过一次完整遍历）
        if self.ptr == 0 and batch_size > 0:
            self.is_full = True
    
    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前记忆库中的所有有效特征和标签。
        
        Returns:
            tuple: (features, labels)
                - features: 形状 (N, D)，其中 N <= K
                - labels: 形状 (N,)
        """
        if self.is_full:
            # 记忆库已满，返回全部
            return self.feats, self.labels
        else:
            # 记忆库未满，只返回已填充的部分
            return self.feats[:self.ptr], self.labels[:self.ptr]
    
    def size(self) -> int:
        """返回记忆库中当前有效样本的数量。"""
        return self.K if self.is_full else self.ptr
    
    def is_empty(self) -> bool:
        """检查记忆库是否为空。"""
        return self.ptr == 0 and not self.is_full


class SupConLossWithXBM(torch.nn.Module):
    """
    支持XBM的监督对比损失。
    
    修改要点:
    1. 接收记忆库作为额外的负样本池
    2. 将当前批次的特征与记忆库特征拼接后计算损失
    3. 记忆库特征不参与梯度计算（已detach）
    """
    
    def __init__(self, temperature: float = 0.07, xbm: Optional[XBM] = None):
        """
        Args:
            temperature (float): 温度超参数
            xbm (Optional[XBM]): XBM记忆库实例（如果为None则退化为普通SupCon）
        """
        super().__init__()
        self.temperature = temperature
        self.xbm = xbm
    
    def forward(self, 
                acoustic_embeddings: torch.Tensor, 
                text_embeddings: Optional[torch.Tensor], 
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算支持XBM的监督对比损失。
        
        Args:
            acoustic_embeddings: 声学嵌入 (B, D)
            text_embeddings: 文本嵌入 (B, D) 或 None
            labels: 标签 (B,)
        
        Returns:
            loss: 标量损失值
        """
        device = acoustic_embeddings.device
        
        # --- 1. 特征归一化 ---
        acoustic_embeddings = F.normalize(acoustic_embeddings, p=2, dim=1)
        
        # --- 2. 构建当前批次的特征矩阵 ---
        if text_embeddings is not None:
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            current_feats = torch.cat([acoustic_embeddings, text_embeddings], dim=0)
            current_labels = labels.repeat(2)
        else:
            current_feats = acoustic_embeddings
            current_labels = labels
        
        batch_size = current_feats.size(0)
        
        # --- 3. [关键] 融合记忆库特征 ---
        if self.xbm is not None and not self.xbm.is_empty():
            # 从记忆库获取历史特征（已经是detached的）
            memory_feats, memory_labels = self.xbm.get()
            
            # 拼接：[当前批次 + 记忆库]
            all_feats = torch.cat([current_feats, memory_feats], dim=0)
            all_labels = torch.cat([current_labels, memory_labels], dim=0)
        else:
            # 记忆库为空或未启用，退化为普通SupCon
            all_feats = current_feats
            all_labels = current_labels
        
        # --- 4. 构造正样本掩码 ---
        # 注意：mask 的形状是 (batch_size, all_feats.size(0))
        # 我们只为当前批次的样本计算损失，但可以与记忆库中的样本对比
        mask = torch.eq(
            current_labels.unsqueeze(1),  # (batch_size, 1)
            all_labels.unsqueeze(0)       # (1, batch_size + K)
        ).float().to(device)
        
        # --- 5. 计算相似度矩阵 ---
        # (batch_size, batch_size + K)
        similarity_matrix = torch.matmul(
            current_feats, all_feats.T
        ) / self.temperature
        
        # --- 6. 排除自身对比（只在当前批次内部） ---
        logits_mask = torch.ones_like(similarity_matrix)
        # 对角线位置（样本与自身）设为0
        logits_mask[:, :batch_size].fill_diagonal_(0)
        
        mask = mask * logits_mask
        
        # --- 7. 计算损失 ---
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss


def demo_xbm_usage():
    """演示XBM的基本使用方法"""
    print("=" * 60)
    print("XBM 使用示例")
    print("=" * 60)
    
    # 1. 初始化XBM
    memory_size = 1024  # 实际训练中建议16384
    feat_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    xbm = XBM(memory_size=memory_size, feat_dim=feat_dim, device=device)
    
    # 2. 模拟训练循环
    for step in range(5):
        # 模拟一个批次的特征和标签
        batch_size = 32
        feats = torch.randn(batch_size, feat_dim, device=device)
        feats = F.normalize(feats, p=2, dim=1)  # L2归一化
        labels = torch.randint(0, 4, (batch_size,), device=device)
        
        # 3. 更新记忆库（注意必须detach）
        xbm.enqueue_dequeue(feats.detach(), labels)
        
        print(f"Step {step + 1}: 记忆库大小 = {xbm.size()}/{memory_size}")
    
    # 4. 获取记忆库内容
    memory_feats, memory_labels = xbm.get()
    print(f"\n最终记忆库: 特征形状={memory_feats.shape}, 标签形状={memory_labels.shape}")
    print("=" * 60)


if __name__ == "__main__":
    demo_xbm_usage()
