# contrastive/trainer.py
import torch
import torch.nn as nn
import transformers
import tqdm
import logging
import gc
import os
import numpy as np
import itertools

from core.trainer import AbstractTrainer
from core.config import CONFIG, device
from contrastive.model import ContrastiveModel, MemoryOptimizedContrastiveModel
from contrastive.loss import SupConLoss
from pytorch_metric_learning.losses import NTXentLoss 

logger = logging.getLogger(__name__)

class ContrastiveTrainer(AbstractTrainer):
    """
    双模态对比学习的训练器 (LGCA框架)。
    该训练器继承自 AbstractTrainer，但重写了其核心训练循环，
    以支持 ContrastiveModel 的多输入和多输出，并使用组合损失函数进行训练。
    """
    # def __init__(self, model: ContrastiveModel, num_epochs: int, learning_rate: float,
    def __init__(self, model: MemoryOptimizedContrastiveModel, num_epochs: int, learning_rate: float,
                 alpha: float,  # <--- 新增参数
                 optimizer_type: str = "AdamW", 
                 gradient_accumulation_steps: int = 4):

        # --- 1. 初始化优化器和损失函数 ---
        if optimizer_type.lower() == "adamw":
            # --- 新的差分学习率逻辑 ---
            
            # 1. 定义学习率
            #    从构造函数传入的 learning_rate 作为 backbone_lr
            # backbone_lr = learning_rate 

            #    从 CONFIG 读取新的 head_lr
            head_lr = CONFIG.training_head_lr()

            print(f"--- [优化器配置] Backbone LR: {learning_rate}, Head LR: {head_lr} ---")

            # 2. 将参数分组
            # (这会自动处理您之前可能设置的任何 'requires_grad=False' 冻结层)
            backbone_params = itertools.chain(
                model.audio_encoder.parameters(),
                model.text_encoder.parameters()
            )
            
            head_params = itertools.chain(
                model.audio_projection_head.parameters(),
                model.text_projection_head.parameters(),
                model.audio_classifier.parameters()
            )
            
            # 3. 创建参数组列表
            optimizer_parameters = [
                {"params": backbone_params, "lr": learning_rate},
                {"params": head_params, "lr": head_lr}
            ]

            optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate, weight_decay=CONFIG.weight_decay())
        else: # 默认为 Adam
            optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate, weight_decay=CONFIG.weight_decay())

        # 实例化两个损失函数：监督对比损失L_LGCA 和 交叉熵损失L_SER
        self.sup_con_loss = SupConLoss(temperature=0.1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # # 从配置中获取损失的加权因子 alpha
        # self.alpha = CONFIG.llgca_loss_alpha()
        
        # 不再从CONFIG中读取，而是直接使用传入的参数
        self.alpha = alpha  # <--- 修改之处

        # 调用父类的构造函数，完成优化器、epoch数、eval评估方法和plot_histories可视化方法
        # 注意这里的 loss 参数可以传 None，因为我们使用自定义的组合损失
        super().__init__(
            model=model,
            num_epochs=num_epochs,
            optimizer=optimizer,
            loss=None,  # 我们将手动计算组合损失
            name="Contrastive_LGCA",
            alpha=self.alpha
        )

        # 冻结特征提取器以加速训练并节省内存（可选，但推荐初次训练时使用）
        # if hasattr(self.model, 'audio_encoder') and hasattr(self.model.audio_encoder, 'freeze_feature_extractor'):
        #     self.model.audio_encoder.freeze_feature_extractor()
        #     print("[INFO] WavLM 特征提取层已冻结。")

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.amp.GradScaler(enabled=True) # 用于混合精度训练
        print(f"[INFO] 使用梯度累积步数: {gradient_accumulation_steps}, 损失权重 alpha: {self.alpha}")

    def _get_outputs_and_labels(self, batch: dict):
        """
        一个辅助方法，用于处理输入批次并从模型获取所有输出。
        """
        # 从批次数据中解包音频、文本和标签
        audio_inputs = batch['audio_input_values'].to(device, non_blocking=True)
        text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
        text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 使用混合精度进行前向传播
        with torch.amp.autocast('cuda'):
            acoustic_embedding, text_embedding, audio_logits = self.model(
                audio_input_values=audio_inputs,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )

        return acoustic_embedding, text_embedding, audio_logits, labels

    # --- *** 新增这个用于评估的方法 *** ---
    def _get_logits_and_real(self, batch: dict) -> (torch.Tensor, torch.Tensor):
        """
        为评估阶段实现此方法，以适配 AbstractTrainer 中的 eval 循环。
        该方法遵循“测试时单模态”的原则，只使用声学分支。
        """
        # 1. 从批次中解包音频输入和真实标签
        audio_inputs = batch['audio_input_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # 2. 模型前向传播（只获取 audio_logits）
        # 我们调用模型，但只关心第三个返回值
        _, _, audio_logits = self.model(
            audio_input_values=audio_inputs,
            text_input_ids=None, # <-- 在评估时，文本输入为None
            text_attention_mask=None
        )
        
        # 3. 返回 eval 方法期望的两个值
        return audio_logits, labels
    # ------------------------------------
    
    def train(self, train_dataloader, val_dataloader=None):
        """
        重写核心的训练方法。
        """
        
        total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self._num_epochs
        # 确保 total_steps 至少为1，避免学习率调度器出错
        total_steps = max(1, total_steps)

        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        # history_loss = []
        # history_acc = []

        # 用于存储每个epoch的平均指标
        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []


        logger.info(f"开始训练 {self._name} 模型...")


        for epoch in range(1, self._num_epochs + 1):
    
            # --- 训练循环 (一个 Epoch) ---
            self.model.train()
            epoch_train_losses, epoch_train_accuracies = [], []
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中]")
            # 在epoch开始时，清零一次梯度,应该在更新后清零
            # self._optimizer.zero_grad()

            for step, batch in enumerate(loader):
                try:
                    # --- 2. 前向传播与损失计算 ---
                    acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(batch)

                    with torch.amp.autocast('cuda'):
                        loss_sup_con = self.sup_con_loss(acoustic_embedding, text_embedding, labels) # 计算监督对比损失
                        loss_ce = self.cross_entropy_loss(audio_logits, labels) # 计算交叉熵损失
                        # total_loss = self.alpha * loss_sup_con + (1 - self.alpha) * loss_ce # 根据 alpha 加权组合损失，油梯度冲突
                        total_loss = self.alpha * loss_sup_con + loss_ce # 根据 alpha 加权组合损失
                        # 暂时只使用单个损失
                        # total_loss = loss_sup_con 
                        # total_loss = loss_ce

                        scaled_loss = total_loss / self.gradient_accumulation_steps # 用于梯度累积的损失缩放

                    # --- 3. 反向传播与优化 ---
                    self.scaler.scale(scaled_loss).backward()

                    # 计算准确率 (用于监控)
                    with torch.no_grad():
                        accuracy = (torch.argmax(audio_logits, dim=1) == labels).float().mean()

                    epoch_train_losses.append(total_loss.item())
                    epoch_train_accuracies.append(accuracy.item())

                    # 梯度累积步骤
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        # 1. 梯度裁剪
                        self.scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        # 2. 优化器更新权重
                        self.scaler.step(self._optimizer)
                        self.scaler.update() # 3. 更新 GradScaler
                        scheduler.step() # 4. 更新学习率
                        self._optimizer.zero_grad() # 5. 清空梯度，为下一个累积周期做准备

                        loader.set_postfix(loss=total_loss.item(), acc=accuracy.item(), sup_con=loss_sup_con.item(), ce=loss_ce.item())

                    # 清理内存
                    if step % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

            # 计算并存储该epoch的平均训练指标
            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))


            # --- 验证循环 (一个 Epoch，采用单模态评估) ---
            if val_dataloader:
                self.model.eval()
                epoch_val_losses, epoch_val_accuracies = [], []
                val_loader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch} [验证中]")
                with torch.no_grad():
                    for batch in val_loader:
                        if not batch: continue
                        
                        # acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(batch) #这是双模态前向传播

                        # **使用单模态方法，只获取logits和真实标签**
                        logits, labels = self._get_logits_and_real(batch)
                        
                        # with torch.amp.autocast('cuda'):
                        #     loss_sup_con = self.sup_con_loss(acoustic_embedding, text_embedding, labels)
                        #     loss_ce = self.cross_entropy_loss(audio_logits, labels)
                        #     val_loss = self.alpha * loss_sup_con + loss_ce

                        # preds = torch.argmax(audio_logits, dim=1)
                        # val_accuracy = (preds == labels).float().mean()

                        # **只计算交叉熵损失，因为它直接反映分类性能**
                        val_loss = self.cross_entropy_loss(logits, labels)
                        val_accuracy = (torch.argmax(logits, dim=1) == labels).float().mean()
                        
                        epoch_val_losses.append(val_loss.item())
                        epoch_val_accuracies.append(val_accuracy.item())

                # 计算并存储该epoch的平均验证指标

                history_val_loss.append(np.mean(epoch_val_losses))
                history_val_acc.append(np.mean(epoch_val_accuracies))
                
                logger.info(f"Epoch {epoch} 总结: 训练损失: {history_train_loss[-1]:.4f}, 训练准确率: {history_train_acc[-1]:.4f}, "
                            f"验证损失: {np.mean(epoch_val_losses):.4f}, 验证准确率: {np.mean(epoch_val_accuracies):.4f}")
                
        # print("total_loss = loss_sup_con")
        # print("total_loss = loss_ce")
        # 所有epoch结束后，绘制训练曲线
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)



class AblationTrainer(ContrastiveTrainer):
    """
    LGCA消融实验的训练器 (无标签指导)。
    
    这个训练器继承自ContrastiveTrainer，但做了以下关键修改：
    1. 将监督对比损失 (SupConLoss) 替换为无监督对比损失 (NTXentLoss)。
    2. 修改了训练和验证循环中的损失计算方式，不再向对比损失函数传递标签。
    """
    def __init__(self, model, num_epochs, learning_rate, alpha, 
                 optimizer_type="AdamW", gradient_accumulation_steps=4):
        
        # --- 几乎与父类完全相同，除了损失函数的实例化 ---
        
        # 调用父类的__init__来设置模型、优化器等
        # 我们先传入一个假的loss，然后替换掉我们关心的部分
        super().__init__(model, num_epochs, learning_rate, alpha, 
                         optimizer_type, gradient_accumulation_steps)

        # *** 关键修改：用NTXentLoss替换SupConLoss ***
        print("[INFO] 初始化 AblationTrainer：使用 NTXentLoss (无标签指导)。")
        self.sup_con_loss = NTXentLoss(temperature=0.1) # 替换掉父类中实例化的SupConLoss
        self.name = "Ablation_LGCA_no_Label"
    
    def train(self, train_dataloader, val_dataloader=None):
        """
        重写核心的训练方法以适应NTXentLoss。
        大部分代码与父类相同，只修改损失计算部分。
        """
        
        total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self._num_epochs
        # 确保 total_steps 至少为1，避免学习率调度器出错
        total_steps = max(1, total_steps)

        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )


        # 用于存储每个epoch的平均指标
        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []


        logger.info(f"开始训练 {self._name} 模型...")
        # self.model.train()
        # self._optimizer.zero_grad()

        for epoch in range(1, self._num_epochs + 1):
            self.model.train()
            epoch_train_losses, epoch_train_accuracies = [], []
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中 - Ablation]")
            # self._optimizer.zero_grad()

            for step, batch in enumerate(loader):
                try:
                    # --- 1. 前向传播 (与之前完全一样) ---
                    acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(batch)

                    with torch.amp.autocast('cuda'):
                        # *** 关键修改：调用NTXentLoss，不传入labels ***
                        # 原始调用: self.sup_con_loss(acoustic_embedding, text_embedding, labels)
                        loss_contrastive = self.sup_con_loss(acoustic_embedding, text_embedding)

                        loss_ce = self.cross_entropy_loss(audio_logits, labels)
                        total_loss = self.alpha * loss_contrastive + loss_ce

                        scaled_loss = total_loss / self.gradient_accumulation_steps

                    # --- 2. 反向传播与优化 (与之前完全一样) ---
                    self.scaler.scale(scaled_loss).backward()
                    
                    with torch.no_grad():
                        preds = torch.argmax(audio_logits, dim=1)
                        accuracy = (preds == labels).float().mean()
                    
                    epoch_train_losses.append(total_loss.item())
                    epoch_train_accuracies.append(accuracy.item())

                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self._optimizer)
                        self.scaler.update()
                        scheduler.step() # 如果有scheduler
                        self._optimizer.zero_grad()

                        loader.set_postfix(loss=total_loss.item(), acc=accuracy.item(), 
                                           contrastive=loss_contrastive.item(), ce=loss_ce.item())

                    # 清理内存
                    if step % 50 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

            # 计算并存储该epoch的平均训练指标
            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))


            # --- 验证循环 (一个 Epoch) ---
            if val_dataloader:
                self.model.eval()
                epoch_val_losses, epoch_val_accuracies = [], []
                val_loader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch} [验证中]")
                with torch.no_grad():
                    for batch in val_loader:
                        if not batch: continue
                        # 需要修改
                        acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(batch)
                        
                        with torch.amp.autocast('cuda'):
                            # *** 关键修改：验证集的对比损失计算 ***
                            loss_contrastive_val = self.sup_con_loss(acoustic_embedding, text_embedding) 
                            loss_ce_val = self.cross_entropy_loss(audio_logits, labels)
                            val_loss = self.alpha * loss_contrastive_val + loss_ce_val
                        
                        preds = torch.argmax(audio_logits, dim=1)
                        val_accuracy = (preds == labels).float().mean()
                        
                        epoch_val_losses.append(val_loss.item())
                        epoch_val_accuracies.append(val_accuracy.item())

                # 计算并存储该epoch的平均验证指标
                avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
                avg_val_acc = sum(epoch_val_accuracies) / len(epoch_val_accuracies)
                history_val_loss.append(np.mean(epoch_val_losses))
                history_val_acc.append(np.mean(epoch_val_accuracies))
                
                logger.info(f"Epoch {epoch} 总结: 训练损失: {history_train_loss[-1]:.4f}, 训练准确率: {history_train_acc[-1]:.4f}, "
                            f"验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_acc:.4f}")
                

        print(f"消融模型 {self.name} 训练完成。")
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)
        # self.plot_histories(history_loss, history_acc)