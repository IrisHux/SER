import logging
import torch
import gc
import tqdm
import transformers
from torch.utils.data import DataLoader
import torch.nn as nn

from core.config import device
from core.trainer import AbstractTrainer
from vizualisers.plots import PlotVisualizer
from sklearn.metrics import confusion_matrix
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class MemoryOptimizedAudioBaselineTrainer(AbstractTrainer):
    """
    内存优化版的音频基线训练器
    """
    def __init__(self, model: nn.Module, num_epochs: int, learning_rate: float,
                 optimizer_type: str = "Adam", gradient_accumulation_steps: int = 4):

        # Explicitly convert learning_rate to float to avoid TypeError
        learning_rate = float(learning_rate)

        # 根据配置选择优化器
        if optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

        super().__init__(
            model=model,
            num_epochs=num_epochs,
            optimizer=optimizer,
            loss=nn.CrossEntropyLoss(),
            name="Audio_Baseline_MemOpt"
        )

        # 冻结特征提取器以节省内存
        self.model.wavlm.freeze_feature_extractor()

        # 梯度累积步数 - 用于模拟更大的batch size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        print(f"[INFO] 特征提取层已冻结，使用梯度累积步数: {gradient_accumulation_steps}")

    def _get_logits_and_real(self, batch: dict) -> (torch.Tensor, torch.Tensor):
        """
        从批次数据中获取模型的输入和真实标签，并执行前向传播。
        """
        audio_inputs = batch['audio_input_values']
        labels = batch['labels']

        # 确保数据在正确的设备上
        audio_inputs = audio_inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 处理维度问题
        if audio_inputs.dim() == 3 and audio_inputs.shape[0] == 1:
            audio_inputs = audio_inputs.squeeze(0)

        # 执行前向传播
        with torch.amp.autocast('cuda'):  # 使用混合精度训练
            logits = self.model(audio_inputs)

        return logits, labels

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        """
        内存优化版本的训练方法，包含验证循环
        """
        # 启用混合精度训练
        scaler = torch.amp.GradScaler('cuda')

        total_steps = len(train_dataloader) * self._num_epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        history_loss = []
        history_acc = []
        val_history_loss = []
        val_history_acc = []

        logger.info(f"Training the {self._name} model with memory optimization...")
        self.model.train()

        for epoch in range(1, self._num_epochs + 1):
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")
            accumulated_loss = 0
            accumulated_acc = 0
            epoch_loss = 0
            epoch_acc = 0
            step_count = 0

            for step, batch in enumerate(loader):
                try:
                    # 使用混合精度训练
                    with torch.amp.autocast('cuda'):
                        logits, real = self._get_logits_and_real(batch)
                        loss = self._loss(logits, real) / self.gradient_accumulation_steps

                    # 反向传播
                    scaler.scale(loss).backward()

                    # 计算准确率
                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        accuracy = torch.mean((preds == real).float())

                    accumulated_loss += loss.item()
                    accumulated_acc += accuracy.item()

                    # 梯度累积
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                        # 优化器步骤
                        scaler.step(self._optimizer)
                        scaler.update()
                        self._optimizer.zero_grad()

                        # 记录指标
                        avg_loss = accumulated_loss / self.gradient_accumulation_steps
                        avg_acc = accumulated_acc / self.gradient_accumulation_steps

                        # 记录epoch统计
                        epoch_loss += avg_loss
                        epoch_acc += avg_acc
                        step_count += 1

                        loader.set_postfix(loss=avg_loss, accuracy=avg_acc)

                        # 重置累积器
                        accumulated_loss = 0
                        accumulated_acc = 0

                    # 定期清理内存
                    if step % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[WARNING] 内存不足，跳过该批次: {e}")
                        # 清理内存并继续
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

            # 计算并记录epoch平均值
            if step_count > 0:
                epoch_avg_loss = epoch_loss / step_count
                epoch_avg_acc = epoch_acc / step_count
                history_loss.append(epoch_avg_loss)
                history_acc.append(epoch_avg_acc)
            
            # 每个epoch结束后进行验证
            if val_dataloader is not None:
                val_loss, val_acc = self._validate_epoch(val_dataloader)
                val_history_loss.append(val_loss)
                val_history_acc.append(val_acc)
                print(f"Epoch {epoch} - Train Loss: {epoch_avg_loss:.4f}, Train Acc: {epoch_avg_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch} - Train Loss: {epoch_avg_loss:.4f}, Train Acc: {epoch_avg_acc:.4f}")

            # 每个epoch结束后清理内存
            torch.cuda.empty_cache()
            gc.collect()

        # 调用绘制历史曲线方法
        if val_dataloader is not None:
            self.plot_histories(history_loss, history_acc, val_history_loss, val_history_acc)
        else:
            self.plot_training_histories(history_loss, history_acc)
        
        # 返回训练和验证历史
        return {
            'train_loss': history_loss,
            'train_acc': history_acc,
            'val_loss': val_history_loss,
            'val_acc': val_history_acc
        }

    def _validate_epoch(self, val_dataloader: DataLoader):
        """
        验证一个epoch，计算平均损失和准确率
        """
        self.model.eval()
        total_val_loss = 0
        total_val_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dataloader, desc="Validating", leave=False):
                try:
                    # 使用与训练相同的方法获取logits和标签
                    with torch.amp.autocast('cuda'):
                        logits, labels = self._get_logits_and_real(batch)
                        loss = self._loss(logits, labels)
                    
                    # 计算准确率
                    preds = torch.argmax(logits, dim=1)
                    accuracy = torch.mean((preds == labels).float())
                    
                    total_val_loss += loss.item()
                    total_val_acc += accuracy.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"[WARNING] 验证时内存不足，跳过该批次: {e}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e
        
        # 切换回训练模式
        self.model.train()
        
        # 返回平均损失和准确率
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
        avg_val_acc = total_val_acc / num_batches if num_batches > 0 else 0
        
        return avg_val_loss, avg_val_acc

    def plot_training_histories(self, train_losses: List[float], train_accuracies: List[float]):
        """
        绘制仅训练历史曲线（没有验证数据）
        """
        try:
            # 使用空列表作为验证数据来调用父类方法
            super().plot_histories(train_losses, train_accuracies, [], [])
        except Exception as e:
            print(f"[WARNING] 无法绘制训练历史图表: {e}")
            # 如果绘图失败，至少打印训练统计信息
            if train_losses and train_accuracies:
                print(f"训练完成!")
                print(f"最终训练损失: {train_losses[-1]:.4f}")
                print(f"最终训练准确率: {train_accuracies[-1]:.4f}")
                print(f"平均训练损失: {np.mean(train_losses):.4f}")
                print(f"平均训练准确率: {np.mean(train_accuracies):.4f}")
