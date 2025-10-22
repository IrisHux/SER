# contrastive/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss) 的修正版实现
    源自论文: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, acoustic_embeddings, text_embeddings, labels):
        """
        计算双模态输入的监督对比损失。
        此版本旨在拉近具有相同情感标签的音频和文本嵌入。

        Args:
            acoustic_embeddings (torch.Tensor): 声学模态的嵌入向量。形状: (B, D)
            text_embeddings (torch.Tensor): 文本模态的嵌入向量。形状: (B, D)
            labels (torch.Tensor): 对应的情感标签。形状: (B)

        Returns:
            torch.Tensor: 计算出的损失值 (一个标量)。
        """
        device = acoustic_embeddings.device
        batch_size = acoustic_embeddings.shape[0]

        # --- 1. 特征准备 ---
        # L2归一化
        acoustic_embeddings = nn.functional.normalize(acoustic_embeddings, p=2, dim=1)
        text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=1)

        # --- 2. 计算跨模态相似度矩阵 ---
        # 形状: (B, B)， sim[i, j] 表示 audio_i 和 text_j 的相似度
        similarity_matrix = torch.matmul(acoustic_embeddings, text_embeddings.T) / self.temperature

        # --- 3. 构造正样本对的掩码 (Mask) ---
        # 如果 audio_i 和 text_j 的情感标签相同，则它们是正样本对
        labels_row = labels.unsqueeze(1) # 形状: (B, 1)
        labels_col = labels.unsqueeze(0) # 形状: (1, B)
        positive_mask = torch.eq(labels_row, labels_col).float().to(device)

        # --- 4. 计算损失 ---
        # 我们希望对于每个 audio anchor，其对应的 text positive samples 的概率之和最大化。
        # positive_mask 已经定义了哪些是正样本。
        # CrossEntropyLoss 的标签可以是软标签（概率分布）。
        # 我们将 positive_mask 归一化，使其每一行的和为1，作为目标概率分布。
        target_labels = positive_mask / (positive_mask.sum(dim=1, keepdim=True) + 1e-8)

        # 计算从音频到文本的损失
        # 使用 log_softmax 手动计算交叉熵，以支持软标签
        log_softmax_sim_a2t = F.log_softmax(similarity_matrix, dim=1)
        loss_a2t = -torch.sum(target_labels * log_softmax_sim_a2t, dim=1).mean()

        # 计算从文本到音频的损失 (对称)
        log_softmax_sim_t2a = F.log_softmax(similarity_matrix.T, dim=1)
        loss_t2a = -torch.sum(target_labels.T * log_softmax_sim_t2a, dim=1).mean()

        # 最终损失是两个方向损失的平均值
        loss = (loss_a2t + loss_t2a) / 2

        # [NaN 检查] 如果损失为 NaN 或 Inf，返回一个可训练的小值
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  警告：检测到 NaN/Inf 损失！")
            print(f"   - similarity_matrix 范围: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
            print(f"   - positive_mask 求和: {positive_mask.sum().item()}")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        return loss
    
    
class InfoNCELoss(nn.Module):
    """
    一个简单的、用于无监督对比学习的 InfoNCE (NTXent) 损失函数。
    它接收两个视图 (view1, view2) 作为输入，它们互为正样本。
    """
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, view1, view2):
        """
        输入:
            view1: 第一个视图的嵌入, 形状 [B, D]
            view2: 第二个视图的嵌入, 形状 [B, D]
        输出:
            InfoNCE 损失值
        """
        batch_size = view1.shape[0]
        
        # 1. 将两个视图合并，以便计算所有可能的对
        embeddings = torch.cat([view1, view2], dim=0) # 形状 [2*B, D]
        
        # 2. 计算相似性矩阵
        # (a @ b.T) 等价于计算所有行向量两两之间的点积
        # 我们需要L2归一化来得到余弦相似度
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # 3. 创建标签
        # 我们的目标是：
        # - view1[i] 的正样本是 view2[i] (在合并后的矩阵中，索引为 i+B)
        # - view2[i] 的正样本是 view1[i] (在合并后的矩阵中，索引为 i)
        
        # 创建一个对角线为0的掩码，以移除 (view1[i], view1[i]) 这种"自己和自己"的配对
        mask = torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        # 标签张量
        # 目标：sim_matrix[0] 的正样本在 sim_matrix[B]
        # 目标：sim_matrix[B] 的正样本在 sim_matrix[0]
        labels = torch.cat([
            torch.arange(batch_size, device=sim_matrix.device) + batch_size,
            torch.arange(batch_size, device=sim_matrix.device)
        ], dim=0) # 形状 [2*B], 内容 [B, B+1, ..., 2B-1, 0, 1, ..., B-1]

        # 4. 计算交叉熵损失
        # sim_matrix 的每一行都是一个 "logits" 分布
        # labels 告诉 CrossEntropyLoss 哪一列是 "正确" 的
        loss = self.criterion(sim_matrix, labels)
        return loss
