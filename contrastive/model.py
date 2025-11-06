# contrastive/model.py
import os
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import CONFIG, device
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from transformers import (
    WavLMModel, AutoModel,
    get_linear_schedule_with_warmup,
    AutoTokenizer
)

# ===============================
# 1. 全局内存优化设置
# ===============================
def setup_memory_optimization():
    """设置全局内存优化参数"""
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 启用内存池优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 设置内存碎片整理
    torch.cuda.empty_cache()

    print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("内存优化设置完成")

# 步骤 1: 定义一个可复用的交叉注意力模块
class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: [Batch, SeqLen_Q, Dim] - 发出查询的模态
        # key_value: [Batch, SeqLen_KV, Dim] - 被查询的模态

        # 交叉注意力 + Add & Norm
        attn_output, _ = self.attention(query, key_value, key_value)
        x = self.norm1(query + self.dropout(attn_output))

        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x
# ===============================
# 2. 内存优化的模型配置
# ===============================
class MemoryOptimizedContrastiveModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()

        # 使用更小的模型或者冻结部分层
        self.audio_encoder = WavLMModel.from_pretrained(
            CONFIG.audio_encoder_name(),
            use_safetensors = True
        )
        self.text_encoder = AutoModel.from_pretrained(
            CONFIG.text_encoder_name(),
            use_safetensors = True
        )

        # 冻结部分编码器层以减少训练时的计算量和内存占用
        # 这里我们冻结 WavLM 的 CNN 特征提取器和 DeBERTa 的前6层 Transformer 层
        self._freeze_model_parts(num_text_layers_to_freeze=6)

        # # 启用梯度检查点以节省内存
        # if use_gradient_checkpointing:
        #     self.audio_encoder.gradient_checkpointing_enable()
        #     self.text_encoder.gradient_checkpointing_enable()
        #     print("已启用梯度检查点以节省内存")

        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size

        # --- 2. 为每个模态定义投影头 (Projection Head) ---
        # 投影头是一个简单的多层感知机 (MLP)，它将编码器输出的高维特征，
        # 映射到一个统一的、较低维度的嵌入空间，用于计算对比损失。
        # 投影头的维度配置从 config.yaml 文件中读取。
        proj_config = CONFIG.projection_bridge_config()
        projection_dims = proj_config['hidden_dims']

        self.audio_projection_head = nn.Sequential(
            nn.Linear(audio_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        self.text_projection_head = nn.Sequential(
            nn.Linear(text_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        # self.audio_classifier = nn.Linear(audio_hidden_size, num_labels)
        # --- 3. [新] 门控交叉注意力融合分支 (用于 Loss_CE) ---
        acoustic_dim = self.audio_encoder.config.hidden_size # 768
        text_dim = self.text_encoder.config.hidden_size       # 768
        self.d_model = CONFIG.fusion_dim()                # 768
        self.num_heads = CONFIG.fusion_heads()            # 8
        
        # 维度对齐层 (保持不变)
        self.acoustic_projector = nn.Linear(acoustic_dim, self.d_model) if acoustic_dim != self.d_model else nn.Identity()
        self.text_projector = nn.Linear(text_dim, self.d_model) if text_dim != self.d_model else nn.Identity()

        # [新] 定义声学 "查询" 文本的交叉注意力层
        # 我们只需要一个方向：声学(Ha) "查询" 文本(Hb)
        self.audio_queries_text_attention = CrossAttentionLayer(self.d_model, self.num_heads)

        # [新] 最终的分类器
        # 注意：输入维度现在只是 d_model (768)，而不是 d_model * 2
        # 因为我们是 H_a + Fused_T，而不是 Concat(H_a, Fused_T)
        self.final_classifier = nn.Linear(self.d_model, num_labels)

    def _freeze_model_parts(self, num_text_layers_to_freeze: int):
        """
        冻结编码器的特定部分。
        """
        # --- 1. 冻结 WavLM 的 CNN 特征提取器 ---
        print("--- 正在冻结 WavLM 的 CNN Feature Extractor ---")
        for param in self.audio_encoder.feature_extractor.parameters():
            param.requires_grad = False

        # --- 2. 冻结 DeBERTa 的 Embedding 层 ---
        print("--- 正在冻结 DeBERTa 的 Embeddings ---")
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

        # --- 3. 冻结 DeBERTa 的底层 Transformer ---
        print(f"--- 正在冻结 DeBERTa 的 Encoder 前 {num_text_layers_to_freeze} 层 ---")
        total_layers = len(self.text_encoder.encoder.layer)
        layers_to_freeze = min(num_text_layers_to_freeze, total_layers)

        if layers_to_freeze < num_text_layers_to_freeze:
                print(f"[警告] 想要冻结 {num_text_layers_to_freeze} 层, "
                      f"但模型只有 {total_layers} 层。将只冻结 {layers_to_freeze} 层。")

        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        print("--- 编码器冻结操作完成 ---")

    def masked_mean(self, tensor, mask):
        # x: [B, L, D], mask: [B, L]
        mask = mask.unsqueeze(-1).type_as(tensor)
        return (tensor * mask).sum(1) / mask.sum(1).clamp_min(1e-6)

    def forward(self, audio_input_values, text_input_ids, text_attention_mask, use_text_modality=True):
        # [修复 NaN] 移除模型内部的 autocast，避免与训练器中的 autocast 嵌套
        # ==========================================================
        # 阶段 1: 独立编码
        # ==========================================================
        # Ha_sequence: [Batch, SeqLen_A, 768]
        Ha_sequence = self.audio_encoder(input_values=audio_input_values).last_hidden_state

        # 音频分支 - 使用更小的序列长度
        # audio_outputs = self.audio_encoder(input_values=audio_input_values)
        # 使用池化而不是平均，减少计算量
        pooled_Ha = torch.mean(Ha_sequence, dim=1)
        # 投影和分类
        acoustic_embedding = self.audio_projection_head(pooled_Ha)
        # final_logits = self.audio_classifier(pooled_Ha)

        # 文本分支
        text_embedding = None
        Hb_sequence = None
        if text_input_ids is not None:
            Hb_sequence = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            ).last_hidden_state
            pooled_Hb = self.masked_mean(Hb_sequence, text_attention_mask)
            # text_features = torch.mean(text_outputs.last_hidden_state, dim=1)
            text_embedding = self.text_projection_head(pooled_Hb)

        # ==========================================================
        # [并行分支 B] 门控交叉注意力分类
        # ==========================================================
        
        # 1. 维度对齐 (只对 Ha)
        Ha_proj_sequence = self.acoustic_projector(Ha_sequence) # -> [B, SeqLen_A, 768]

        # 2. 计算融合特征 H
        if use_text_modality and Hb_sequence is not None:
            # 训练阶段 (λ=1)
            Hb_proj_sequence = self.text_projector(Hb_sequence) # -> [B, SeqLen_T, 768]
            
            # 计算“修正”信息
            # Ha 作为 Query，去 Hb 中提取信息
            correction_features = self.audio_queries_text_attention(
                query=Ha_proj_sequence, 
                key_value=Hb_proj_sequence
            )
            
            # H = Ha + λ * CrossAttn(...)
            fused_sequence = Ha_proj_sequence + correction_features
        else:
            # 推理阶段 (λ=0)
            fused_sequence = Ha_proj_sequence # H = Ha
        
        # 3. 池化
        pooled_fused_features = torch.mean(fused_sequence, dim=1)

        # 4. 分类
        final_logits = self.final_classifier(pooled_fused_features)
        
        # 返回4个值:
        # 1. acoustic_embedding: 对比学习的声学嵌入 (投影头输出)
        # 2. text_embedding: 对比学习的文本嵌入 (投影头输出)
        # 3. final_logits: 分类logits
        # 4. pooled_fused_features: 门控融合后的分类特征 (用于可视化)
        return acoustic_embedding, text_embedding, final_logits, pooled_fused_features

    

class AcousticSupConModel(nn.Module):
    """
    消融模型B: LGCA w/o Text Anchor
    一个纯声学的监督对比学习模型。
    """
    def __init__(self, num_labels: int):
        super().__init__()

        # --- 1. 只实例化声学编码器 ---
        self.audio_encoder = WavLMModel.from_pretrained(
            CONFIG.audio_encoder_name(),
            use_safetensors=True
        )

        # (可选但推荐) 冻结特征提取器
        if hasattr(self.audio_encoder, 'feature_extractor'):
            for param in self.audio_encoder.feature_extractor.parameters():
                param.requires_grad = False

        audio_hidden_size = self.audio_encoder.config.hidden_size

        # --- 2. 只需要一个声学投影头 ---
        proj_config = CONFIG.projection_bridge_config()
        projection_dims = proj_config['hidden_dims']
        self.audio_projection_head = nn.Sequential(
            nn.Linear(audio_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        # --- 3. 仍然需要分类头 ---
        self.audio_classifier = nn.Linear(audio_hidden_size, num_labels)

    def forward(self, audio_input_1):
        """
        前向传播接收两个增强版本的音频输入。
        """
        # --- 分别处理两个音频输入 ---
        # 它们共享同一个编码器和投影头
        outputs_1 = self.audio_encoder(input_values=audio_input_1)
        features_1 = torch.mean(outputs_1.last_hidden_state, dim=1)
        embedding_1 = self.audio_projection_head(features_1)
        # --- 计算分类 Logits ---
        # 通常只使用一个视图（例如第一个）的特征来做分类，以保持稳定性
        logits = self.audio_classifier(features_1)

        return embedding_1, logits