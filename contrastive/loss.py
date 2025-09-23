# contrastive/loss.py
import torch
import torch.nn as nn

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