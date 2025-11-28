"""
简单的双模态融合模型 (用于 Stage 1 Step 2)

这个模型实现了最简单的音频-文本融合策略：
1. 分别提取音频和文本特征
2. 使用简单的拼接或加权平均融合
3. 通过分类头进行情感分类

不包含复杂组件（对比学习、交叉注意力、XBM等）
"""

import torch
import torch.nn as nn
from transformers import WavLMModel, AutoModel
from core.config import CONFIG


class SimpleMultimodalModel(nn.Module):
    """
    简单的双模态融合模型
    
    架构:
    1. Audio Encoder (WavLM) -> audio_embedding
    2. Text Encoder (DeBERTa) -> text_embedding
    3. Fusion (concatenation or weighted average)
    4. Classification Head
    """
    
    def __init__(
        self,
        num_labels: int,
        fusion_type: str = "concat",  # "concat", "weighted_avg", "gated"
        freeze_encoders: bool = False
    ):
        """
        Args:
            num_labels: 情感类别数量
            fusion_type: 融合方式
                - "concat": 简单拼接
                - "weighted_avg": 可学习的加权平均
                - "gated": 门控融合
            freeze_encoders: 是否冻结编码器（只训练融合层和分类头）
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # 1. 音频编码器 (WavLM)
        self.audio_encoder = WavLMModel.from_pretrained(
            CONFIG.audio_encoder_name(),
            use_safetensors=True
        )
        audio_hidden_size = self.audio_encoder.config.hidden_size  # 768
        
        # 2. 文本编码器 (DeBERTa-V3，使用 AutoModel 自动适配)
        self.text_encoder = AutoModel.from_pretrained(
            CONFIG.text_encoder_name(),
            use_safetensors=True
        )
        text_hidden_size = self.text_encoder.config.hidden_size  # 768
        
        # 可选：冻结编码器
        if freeze_encoders:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("[INFO] 编码器已冻结")
        
        # 3. 融合层
        if fusion_type == "concat":
            # 简单拼接：[audio_emb; text_emb]
            fusion_dim = audio_hidden_size + text_hidden_size  # 1536
            self.fusion = None  # 不需要额外的融合层
            
        elif fusion_type == "weighted_avg":
            # 可学习的加权平均：alpha * audio + (1-alpha) * text
            fusion_dim = audio_hidden_size  # 假设两者维度相同
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5
            self.fusion = None
            
        elif fusion_type == "gated":
            # 门控融合：使用门控机制决定每个模态的重要性
            fusion_dim = audio_hidden_size
            self.audio_gate = nn.Linear(audio_hidden_size, 1)
            self.text_gate = nn.Linear(text_hidden_size, 1)
            self.fusion = None
            
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_labels)
        )
    
    def forward(
        self,
        audio_input_values,
        text_input_ids,
        text_attention_mask
    ):
        """
        前向传播
        
        Args:
            audio_input_values: 音频输入 [batch, audio_len]
            text_input_ids: 文本输入 [batch, text_len]
            text_attention_mask: 文本注意力掩码 [batch, text_len]
            
        Returns:
            logits: 分类输出 [batch, num_labels]
        """
        # 1. 提取音频特征
        audio_outputs = self.audio_encoder(input_values=audio_input_values)
        audio_embedding = audio_outputs.last_hidden_state.mean(dim=1)  # [batch, 768]
        
        # 2. 提取文本特征
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_embedding = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token [batch, 768]
        
        # 3. 融合
        if self.fusion_type == "concat":
            # 简单拼接
            fused = torch.cat([audio_embedding, text_embedding], dim=-1)  # [batch, 1536]
            
        elif self.fusion_type == "weighted_avg":
            # 加权平均（使用sigmoid确保权重在0-1之间）
            alpha = torch.sigmoid(self.fusion_weight)
            fused = alpha * audio_embedding + (1 - alpha) * text_embedding  # [batch, 768]
            
        elif self.fusion_type == "gated":
            # 门控融合
            audio_gate = torch.sigmoid(self.audio_gate(audio_embedding))  # [batch, 1]
            text_gate = torch.sigmoid(self.text_gate(text_embedding))  # [batch, 1]
            
            # 归一化门控权重
            gate_sum = audio_gate + text_gate
            audio_weight = audio_gate / gate_sum
            text_weight = text_gate / gate_sum
            
            fused = audio_weight * audio_embedding + text_weight * text_embedding  # [batch, 768]
        
        # 4. 分类
        logits = self.classifier(fused)  # [batch, num_labels]
        
        return logits


class SimpleMultimodalTrainer:
    """
    简单双模态模型的训练器
    """
    
    def __init__(
        self,
        model: SimpleMultimodalModel,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4
    ):
        from core.config import device
        
        self.model = model.to(device)
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 设备
        self.device = device
    
    def _get_logits_and_labels(self, batch):
        """从batch中提取数据并进行前向传播"""
        # 获取输入
        audio_inputs = batch['audio_input_values'].to(self.device)
        text_input_ids = batch['text_input_ids'].to(self.device)
        text_attention_mask = batch['text_attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 处理音频维度
        if audio_inputs.dim() == 3 and audio_inputs.shape[0] == 1:
            audio_inputs = audio_inputs.squeeze(0)
        
        # 前向传播
        with torch.amp.autocast('cuda'):
            logits = self.model(
                audio_input_values=audio_inputs,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )
        
        return logits, labels
    
    def train(self, train_loader, val_loader=None):
        """训练模型"""
        import transformers
        from tqdm import tqdm
        
        # 混合精度训练
        scaler = torch.amp.GradScaler('cuda')
        
        # 学习率调度器
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            total_loss = 0
            correct = 0
            total = 0
            
            self.optimizer.zero_grad()
            
            progress_bar = tqdm(train_loader, desc=f"Training")
            
            for step, batch in enumerate(progress_bar):
                # 前向传播
                logits, labels = self._get_logits_and_labels(batch)
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad()
                
                # 统计
                total_loss += loss.item() * self.gradient_accumulation_steps
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                progress_bar.set_postfix({
                    'loss': total_loss / (step + 1),
                    'acc': correct / total
                })
            
            # 验证
            if val_loader:
                val_uar, val_war = self.evaluate(val_loader)
                print(f"Validation - UAR: {val_uar:.4f}, WAR: {val_war:.4f}")
    
    def evaluate(self, dataloader):
        """评估模型"""
        from sklearn.metrics import recall_score, accuracy_score
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                logits, labels = self._get_logits_and_labels(batch)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        uar = recall_score(all_labels, all_predictions, average='macro')
        war = accuracy_score(all_labels, all_predictions)
        
        self.model.train()
        return uar, war
