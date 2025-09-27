# 文件名: contrastive/trainer_ablation_no_label.py

import torch
import torch.nn as nn
import transformers
import tqdm
import logging
import gc
import numpy as np

# 导入原始的训练器以便继承
from contrastive.trainer import ContrastiveTrainer
# 从库中导入无监督对比损失函数
from pytorch_metric_learning.losses import NTXentLoss 
from core.config import device

logger = logging.getLogger(__name__)

# 继承自 ContrastiveTrainer 以复用 __init__, eval 等方法
class AblationNoLabelTrainer(ContrastiveTrainer):
    """
    用于 "LGCA w/o Label-Guidance" 消融实验的训练器。
    
    这个训练器用无监督对比损失 (NTXentLoss, InfoNCE 的一种实现)
    替换了监督对比损失 (SupConLoss)。
    """
    def __init__(self, model, num_epochs, learning_rate, alpha, optimizer_type, gradient_accumulation_steps):
        
        # 调用父类的 __init__ 方法来设置模型、优化器等
        # 我们传入 loss=None，因为我们将重写损失计算逻辑
        super().__init__(model, num_epochs, learning_rate, alpha, optimizer_type, gradient_accumulation_steps)
        
        # --- 核心改动: 实例化无监督损失函数 ---
        # 我们使用 NTXentLoss 替代 SupConLoss。这个损失函数不需要情感标签。
        # 它将来自同一样本的 audio-text 对视为正样本对，其他所有样本都视为负样本。
        print("--- [消融训练器初始化] 使用 NTXentLoss (InfoNCE) 进行无监督对比学习 ---")
        self.contrastive_loss = NTXentLoss(temperature=0.1)
        
        # 我们仍然需要交叉熵损失来训练分类头
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # 重写模型名称，以便正确保存模型
        self._name = "LGCA_Ablation_No_Label"
        self._alpha = alpha

    def train(self, train_dataloader, val_dataloader=None):
        """
        重写的训练方法。
        核心逻辑与父类相同，但损失计算部分被修改。
        """
        total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self._num_epochs
        total_steps = max(1, total_steps)

        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []

        logger.info(f"开始训练 {self._name} 模型...")

        for epoch in range(1, self._num_epochs + 1):
            self.model.train()
            epoch_train_losses, epoch_train_accuracies = [], []
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中 - 无标签消融]")
            
            for step, batch in enumerate(loader):
                try:
                    # --- 1. 获取模型输出 (这部分不变) ---
                    acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(batch)

                    # --- 2. 核心改动: 计算损失 ---
                    with torch.amp.autocast('cuda'):
                        # 分类损失的计算方式保持不变
                        loss_ce = self.cross_entropy_loss(audio_logits, labels)

                        # 对比损失的计算方式现在不同了
                        # 我们将两种模态的嵌入向量合并成一个张量，以输入给损失函数
                        # 前半部分 (声学) 将与后半部分 (文本) 配对
                        embeddings_for_loss = torch.cat([acoustic_embedding, text_embedding], dim=0)
                        
                        # 我们需要为 NTXentLoss 创建特殊的 "标签" 来指明正样本对
                        # 这里的标签就是 [0, 1, ..., B-1, 0, 1, ..., B-1]，
                        # 它告诉损失函数 acoustic_embedding[i] 和 text_embedding[i] 是一对正样本
                        batch_size = acoustic_embedding.shape[0]
                        ntxent_labels = torch.arange(batch_size, device=device)
                        ntxent_labels = torch.cat([ntxent_labels, ntxent_labels], dim=0)
                        
                        loss_contrastive = self.contrastive_loss(embeddings_for_loss, ntxent_labels)

                        # 总损失仍然是加权和
                        total_loss = self._alpha * loss_contrastive + loss_ce
                        
                        scaled_loss = total_loss / self.gradient_accumulation_steps

                    # --- 3. 反向传播与优化 (这部分不变) ---
                    self.scaler.scale(scaled_loss).backward()
                    
                    with torch.no_grad():
                        accuracy = (torch.argmax(audio_logits, dim=1) == labels).float().mean()

                    epoch_train_losses.append(total_loss.item())
                    epoch_train_accuracies.append(accuracy.item())

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self._optimizer)
                        self.scaler.update()
                        scheduler.step()
                        self._optimizer.zero_grad()
                        
                        loader.set_postfix(loss=total_loss.item(), acc=accuracy.item(), contrastive=loss_contrastive.item(), ce=loss_ce.item())

                    if step % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

            # 验证循环和绘图部分被完美继承，无需改动
            # 计算并存储该 epoch 的平均训练指标
            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))

            if val_dataloader:
                # 父类的 eval 方法在这里被自动调用
                # 它正确地使用了单模态评估，所以无需修改
                val_metrics = self.eval(val_dataloader, return_dict=True) 
                history_val_loss.append(val_metrics['loss'])
                history_val_acc.append(val_metrics['accuracy'])
                
                logger.info(f"Epoch {epoch} 总结: 训练损失: {history_train_loss[-1]:.4f}, 训练准确率: {history_train_acc[-1]:.4f}, "
                            f"验证损失: {val_metrics['loss']:.4f}, 验证准确率: {val_metrics['accuracy']:.4f}")
        
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)