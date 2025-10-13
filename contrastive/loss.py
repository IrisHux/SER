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

        # --- 2. 修正：将特征和标签沿批次维度拼接 ---
        # 之前的方法是交错拼接，现在改为顺序拼接，逻辑更清晰
        # 形成一个 (2*B, D) 的大批次特征矩阵
        features = torch.cat([acoustic_embeddings, text_embeddings], dim=0)

        # 对应地，标签也需要复制和拼接，以匹配新的特征矩阵
        # [label_1, label_2] -> [label_1, label_2, label_1, label_2]
        labels = labels.repeat(2)

        # --- 3. 构造正样本对的“标签掩码” (Mask) ---
        # 现在 mask 的形状是 (2*B, 2*B)，能正确反映所有样本间的关系
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(device)

        # --- 4. 计算相似度矩阵 ---
        # 这部分逻辑不变
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # --- 5. 构造用于计算损失的掩码 ---
        # 排除对角线（样本与自身的比较）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # --- 6. 计算最终损失 ---
        # 这部分逻辑不变
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # (mask.sum(1)) 可能为0（如果没有其他正样本），为避免除以0，加上一个极小值
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 对整个批次的损失求平均
        loss = - mean_log_prob_pos.mean()

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
