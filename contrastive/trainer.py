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
from contrastive.model import ContrastiveModel, MemoryOptimizedContrastiveModel, AcousticSupConModel
from contrastive.loss import SupConLoss, InfoNCELoss

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

        augmented_audio_inputs = batch.get('augmented_audio_input_values')
        if augmented_audio_inputs is not None:
            augmented_audio_inputs = augmented_audio_inputs.to(device, non_blocking=True)

        # 使用混合精度进行前向传播
        with torch.amp.autocast('cuda'):
            acoustic_embedding, text_embedding, audio_logits, augmented_acoustic_embedding = self.model(
                audio_input_values=audio_inputs,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                augmented_audio_input_values=augmented_audio_inputs
            )

        return acoustic_embedding, text_embedding, audio_logits, augmented_acoustic_embedding, labels

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
        _, _, audio_logits, _ = self.model(
            audio_input_values=audio_inputs,
            text_input_ids=None, # <-- 在评估时，文本输入为None
            text_attention_mask=None
        )
        
        # 3. 返回 eval 方法期望的两个值
        return audio_logits, labels
    # ------------------------------------
    
    def _save_best_model_if_improved(self, val_uar: float, epoch: int):
        """
        如果当前验证UAR超过历史最佳值，则保存模型。
        
        Args:
            val_uar (float): 当前epoch的验证UAR
            epoch (int): 当前epoch数
            min_epoch (int): 开始保存模型的最小epoch数，默认为12
        """
        if not hasattr(self, 'best_uar') or val_uar > self.best_uar:
            self.best_uar = val_uar
            save_path = os.path.join(CONFIG.saved_ckpt_location(), f'{self._name}_model_epoch_{epoch}.pt')
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"新的最佳UAR模型已保存: {val_uar:.4f} (Epoch {epoch}) -> {save_path}")
    
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
                    acoustic_embedding, text_embedding, audio_logits, augmented_acoustic_embedding, labels = self._get_outputs_and_labels(batch)

                    with torch.amp.autocast('cuda'):
                        loss_inter_contrastive = self.sup_con_loss(acoustic_embedding, text_embedding, labels) # 计算监督对比损失
                        loss_ce = self.cross_entropy_loss(audio_logits, labels) # 计算交叉熵损失

                        loss_intra_contrastive = 0
                        if augmented_acoustic_embedding is not None:
                            loss_intra_contrastive = self.sup_con_loss(acoustic_embedding, augmented_acoustic_embedding, labels)

                        total_loss = self.alpha * (loss_inter_contrastive + loss_intra_contrastive) + loss_ce # 根据 alpha 加权组合损失
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

                        # 为了让显示的数值更直观，我们只取损失张量的数值部分 (.item())
                        # 注意 loss_intra_contrastive 可能是一个 float 0，所以不需要 .item()
                        loss_intra_val = loss_intra_contrastive.item() if torch.is_tensor(loss_intra_contrastive) else loss_intra_contrastive

                        loader.set_postfix(
                            loss=total_loss.item(), 
                            acc=accuracy.item(), 
                            ce=loss_ce.item(), 
                            inter_con=loss_inter_contrastive.item(), # 跨模态对比损失
                            intra_con=loss_intra_val              # 模态内对比损失
                        )

                    # 清理内存
                    if step % 20 == 0:
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
                all_preds, all_labels = [], []  # 收集所有预测和标签用于计算UAR
                val_loader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch} [验证中]")
                with torch.no_grad():
                    for batch in val_loader:
                        if not batch: continue
                        
                        # acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(batch) #这是双模态前向传播

                        # **使用单模态方法，只获取logits和真实标签**
                        logits, labels = self._get_logits_and_real(batch)

                        # **只计算交叉熵损失，因为它直接反映分类性能**
                        val_loss = self.cross_entropy_loss(logits, labels)
                        preds = torch.argmax(logits, dim=1)
                        val_accuracy = (preds == labels).float().mean()
                        
                        epoch_val_losses.append(val_loss.item())
                        epoch_val_accuracies.append(val_accuracy.item())

                        # 收集预测和标签
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                # 计算并存储该epoch的平均验证指标
                history_val_loss.append(np.mean(epoch_val_losses))
                history_val_acc.append(np.mean(epoch_val_accuracies))
                
                # 计算UAR (非加权平均召回率)
                from sklearn.metrics import recall_score
                val_uar = recall_score(all_labels, all_preds, average='macro')
                
                logger.info(f"Epoch {epoch} 总结: 训练损失: {history_train_loss[-1]:.4f}, 训练准确率: {history_train_acc[-1]:.4f}, "
                            f"验证损失: {np.mean(epoch_val_losses):.4f}, 验证准确率: {np.mean(epoch_val_accuracies):.4f}, "
                            f"验证UAR: {val_uar:.4f}")
                
                # 保存最佳UAR模型
                # if epoch >= 12:
                self._save_best_model_if_improved(val_uar, epoch)

        # 所有epoch结束后，绘制训练曲线
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)



class AblationNoLabelTrainer(ContrastiveTrainer):
    """
    消融模型 A (LGCA w/o Label-Guidance) 的训练器。
    继承自 ContrastiveTrainer，但使用无监督对比损失 (InfoNCE/NTXentLoss)
    替代了监督对比损失 (SupConLoss)。
    """
    def __init__(self, model: MemoryOptimizedContrastiveModel, num_epochs: int, learning_rate: float,
                 alpha: float,
                 optimizer_type: str = "AdamW", 
                 gradient_accumulation_steps: int = 4):
        
        # 1. 调用父类的 __init__ 来完成所有基础设置 (优化器, 学习率, etc.)
        # 注意，父类的 __init__ 会创建一个我们不会使用的 self.sup_con_loss
        super().__init__(model, num_epochs, learning_rate, alpha, optimizer_type, gradient_accumulation_steps)

        # 2. [修改] 将损失函数替换为无监督的 NTXentLoss
        print("--- [损失函数修改] 使用无监督对比损失 InfoNCELoss ---")
        self.contrastive_loss = InfoNCELoss(temperature=0.1)
        # 交叉熵损失 self.cross_entropy_loss 保持不变，已在父类中创建
        
        # 3. [修改] 更新模型名称，用于保存和绘图
        self._name = "Ablation_LGCA_no_Label"
        self._alpha = alpha
        
    def train(self, train_dataloader, val_dataloader=None):
        """
        重写核心的训练方法以适应无监督对比损失。
        大部分逻辑与父类相同，仅修改损失计算部分。
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
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中]")

            for step, batch in enumerate(loader):
                try:
                    acoustic_embedding, text_embedding, audio_logits, augmented_acoustic_embedding, labels = self._get_outputs_and_labels(batch)

                    with torch.amp.autocast('cuda'):
                        # 1. 分类损失的计算方式保持不变
                        loss_ce = self.cross_entropy_loss(audio_logits, labels)

                        # 3. [核心修正] 计算无监督对比损失
                        #    我们的新 InfoNCELoss 期望 (view1, view2)
                        
                        # a. 跨模态损失 (Audio-Text)
                        #    直接传入两个视图，不再需要手动创建标签
                        loss_inter_contrastive = self.contrastive_loss(acoustic_embedding, text_embedding)
                        
                        # b. 模态内损失 (Audio-Audio_Augmented)
                        loss_intra_contrastive = 0
                        if augmented_acoustic_embedding is not None:
                            # 同样直接传入两个视图
                            loss_intra_contrastive = self.contrastive_loss(acoustic_embedding, augmented_acoustic_embedding)                        
                        # 3. 计算总损失，保持公平的结构
                        total_loss = self.alpha * (loss_inter_contrastive + loss_intra_contrastive) + loss_ce
                        scaled_loss = total_loss / self.gradient_accumulation_steps

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
                        
                        # 更新进度条显示
                          # [核心修改] 更新进度条显示以反映新的损失分量
                        loss_intra_val = loss_intra_contrastive.item() if torch.is_tensor(loss_intra_contrastive) else loss_intra_contrastive
                        loader.set_postfix(
                            loss=total_loss.item(), acc=accuracy.item(), ce=loss_ce.item(), 
                            inter_con=loss_inter_contrastive.item(), intra_con=loss_intra_val
                        )

                    if step % 20 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))

            # 验证循环与父类完全相同，所以我们直接调用父类的逻辑
            # 这里我们手动实现，因为我们重写了整个 train 方法
            if val_dataloader:
                self.model.eval()
                epoch_val_losses, epoch_val_accuracies = [], []
                all_preds, all_labels = [], []
                val_loader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch} [验证中]")
                with torch.no_grad():
                    for batch in val_loader:
                        if not batch: continue
                        logits, labels = self._get_logits_and_real(batch)
                        val_loss = self.cross_entropy_loss(logits, labels)
                        preds = torch.argmax(logits, dim=1)
                        val_accuracy = (preds == labels).float().mean()
                        
                        epoch_val_losses.append(val_loss.item())
                        epoch_val_accuracies.append(val_accuracy.item())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                history_val_loss.append(np.mean(epoch_val_losses))
                history_val_acc.append(np.mean(epoch_val_accuracies))
                
                from sklearn.metrics import recall_score
                val_uar = recall_score(all_labels, all_preds, average='macro')
                
                logger.info(f"Epoch {epoch} 总结: 训练损失: {history_train_loss[-1]:.4f}, 训练准确率: {history_train_acc[-1]:.4f}, "
                            f"验证损失: {np.mean(epoch_val_losses):.4f}, 验证准确率: {np.mean(epoch_val_accuracies):.4f}, "
                            f"验证UAR: {val_uar:.4f}")
                
                self._save_best_model_if_improved(val_uar, epoch)

        # 调用父类的绘图方法
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)



class AblationNoTextTrainer(ContrastiveTrainer):
    """
    [已修正] 消融模型B (LGCA w/o Text Anchor) 的专用训练器。

    修正内容:
    - 重写了 __init__ 方法，为纯声学模型定制了优化器参数分组，
      解决了因调用父类__init__而导致的 AttributeError。
    """

    # 步骤 1: [核心修正] 为子类重写 __init__ 方法
    def __init__(self, model: AcousticSupConModel, num_epochs: int, learning_rate: float,
                 alpha: float, optimizer_type: str = "AdamW",
                 gradient_accumulation_steps: int = 4):

        # --- 1. 为纯声学模型定制优化器参数分组 ---
        # 这里的逻辑与父类相似，但移除了所有与 text_encoder 相关的部分
        if optimizer_type.lower() == "adamw":
            head_lr = CONFIG.training_head_lr()
            print(f"--- [消融优化器配置] Backbone LR: {learning_rate}, Head LR: {head_lr} ---")

            # `backbone` 只包含 audio_encoder
            backbone_params = model.audio_encoder.parameters()

            # `head` 只包含 audio_projection_head 和 audio_classifier
            head_params = itertools.chain(
                model.audio_projection_head.parameters(),
                model.audio_classifier.parameters()
            )

            optimizer_parameters = [
                {"params": backbone_params, "lr": learning_rate},
                {"params": head_params, "lr": head_lr}
            ]
            optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate, weight_decay=CONFIG.weight_decay())
        else:
            # 为其他优化器类型提供一个简单的（非差分学习率）备用方案
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=CONFIG.weight_decay())

        # --- 2. 初始化损失函数 (这部分逻辑与父类相同) ---
        self.sup_con_loss = SupConLoss(temperature=0.1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

        # --- 3. 调用 *祖父类* (AbstractTrainer) 的 __init__ ---
        # 我们需要手动完成父类 `ContrastiveTrainer` 本来会做的所有初始化工作。
        # 注意这里我们不再调用 super().__init__(...)，因为它会指向有问题的父类方法。
        # 我们直接调用 `AbstractTrainer` 的构造函数。
        # (假设你的 AbstractTrainer 的 __init__ 签名是这样的)
        super(ContrastiveTrainer, self).__init__(
            model=model,
            num_epochs=num_epochs,
            optimizer=optimizer,
            loss=None,  # 我们手动计算组合损失
            name="Ablation_LGCA_no_Text",
            alpha=self.alpha # 如果 AbstractTrainer 接收这个参数
        )

        # --- 4. 初始化训练器状态 (这部分逻辑也来自父类) ---
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = torch.amp.GradScaler(enabled=True)
        print(f"[INFO] 初始化 {self._name} 训练器完成。梯度累积: {gradient_accumulation_steps}, Alpha: {self.alpha}")

    # _get_outputs_and_labels, _get_logits_and_real, train 方法保持不变...
    # (这里省略之前已经提供的其他方法代码，它们是正确的，无需修改)
    def _get_outputs_and_labels(self, batch: dict):
        """
        [核心修改] 重写此辅助方法，以处理新的双音频输入批次。
        这个方法是连接 Dataloader 和 Model 的关键桥梁。
        """
        # 从批次中解包两个增强后的音频视图和标签
        audio_inputs_1 = batch['audio_input_1'].to(device, non_blocking=True)
        audio_inputs_2 = batch['audio_input_2'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 使用混合精度进行前向传播
        # self.model 在这里是 AcousticSupConModel 的一个实例
        with torch.amp.autocast('cuda'):
            embedding_1, embedding_2, audio_logits = self.model(
                audio_input_1=audio_inputs_1,
                audio_input_2=audio_inputs_2
            )

        return embedding_1, embedding_2, audio_logits, labels

    # 步骤 3: 重写验证阶段的"桥梁"方法，以适配纯声学模型
    def _get_logits_and_real(self, batch: dict) -> (torch.Tensor, torch.Tensor):
        """
        [核心修改] 为评估阶段重写此方法。
        该方法遵循“测试时单模态”的原则。
        """
        # 1. 从批次中解包 *两个* 音频输入和真实标签
        audio_inputs = batch['audio_input_values'].to(device, non_blocking=True)
        # audio_inputs_2 = batch['audio_input_2'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 2. 模型前向传播（只获取 audio_logits）
        #    我们调用模型，但只关心第三个返回值。
        #    AcousticSupConModel 的 forward 方法设计为用第一个输入计算 logits。
        _, _, audio_logits = self.model(
            audio_input_1=audio_inputs,
            audio_input_2=None
        )

        # 3. 返回 eval 方法期望的两个值
        return audio_logits, labels

    def train(self, train_dataloader, val_dataloader=None):
        """
        [核心修改] 重写核心训练方法，以适配新的损失计算逻辑。
        大部分代码结构与父类 `ContrastiveTrainer.train` 相同。
        """
        # --- 初始化 (完全复用父类逻辑) ---
        total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self._num_epochs
        total_steps = max(1, total_steps)
        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []
        logger.info(f"开始训练 {self._name} (Ablation w/o Text) 模型...")

        # --- 训练循环 ---
        for epoch in range(1, self._num_epochs + 1):
            self.model.train()
            epoch_train_losses, epoch_train_accuracies = [], []
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中 - No Text Ablation]")

            for step, batch in enumerate(loader):
                try:
                    # 1. 前向传播: 调用我们重写过的 _get_outputs_and_labels
                    embedding_1, embedding_2, audio_logits, labels = self._get_outputs_and_labels(batch)

                    with torch.amp.autocast('cuda'):
                        # 2. 计算损失 [逻辑修改点]
                        #    将两个声学嵌入送入 SupConLoss
                        loss_sup_con = self.sup_con_loss(embedding_1, embedding_2, labels)

                        #    交叉熵损失计算方式不变
                        loss_ce = self.cross_entropy_loss(audio_logits, labels)

                        #    组合损失公式不变
                        total_loss = self.alpha * loss_sup_con + loss_ce

                        scaled_loss = total_loss / self.gradient_accumulation_steps

                    # 3. 反向传播与优化 (完全复用父类逻辑)
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
                        loader.set_postfix(loss=total_loss.item(), acc=accuracy.item(),
                                           sup_con=loss_sup_con.item(), ce=loss_ce.item())

                    if step % 20 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()

            # --- 记录训练历史 (完全复用父类逻辑) ---
            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))

            # --- 验证循环 (完全复用父类逻辑) ---
            if val_dataloader:
                self.model.eval() # 1. 设置为评估模式
                epoch_val_losses, epoch_val_accuracies = [], []
                all_preds, all_labels = [], [] # 用于计算UAR

                # 2. 在 no_grad 上下文中进行
                with torch.no_grad():
                    val_loader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch} [验证中]")
                    for batch in val_loader:
                        if not batch: continue

                        # 3. 获取模型输出
                        #    注意：这里我们只关心分类性能，所以使用 _get_logits_and_real
                        logits, labels = self._get_logits_and_real(batch)

                        # 4. 计算验证损失
                        #    在验证时，只关心分类损失 (CrossEntropy)
                        val_loss = self.cross_entropy_loss(logits, labels)

                        # 5. 计算预测和准确率
                        preds = torch.argmax(logits, dim=1)
                        val_accuracy = (preds == labels).float().mean()

                        # 6. 收集数据用于后续指标计算
                        epoch_val_losses.append(val_loss.item())
                        epoch_val_accuracies.append(val_accuracy.item())
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                # 7. 计算并记录整个 epoch 的平均指标
                avg_val_loss = np.mean(epoch_val_losses)
                avg_val_acc = np.mean(epoch_val_accuracies)
                # UAR 是基于整个验证集的所有预测来计算的
                from sklearn.metrics import recall_score
                val_uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)

                history_val_loss.append(avg_val_loss)
                history_val_acc.append(avg_val_acc)

                logger.info(f"Epoch {epoch} 总结: Train Loss: {history_train_loss[-1]:.4f}, Train Acc: {history_train_acc[-1]:.4f}, "
                            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val UAR: {val_uar:.4f}")

                # 保存最佳模型
                # if epoch >= 12:
                self._save_best_model_if_improved(val_uar, epoch)

        # --- 绘图 (完全复用父类逻辑) ---
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)