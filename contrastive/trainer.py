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
from contrastive.model import MemoryOptimizedContrastiveModel, AcousticSupConModel
from contrastive.loss import SupConLoss, InfoNCELoss
from contrastive.xbm import XBM, SupConLossWithXBM

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
                 gradient_accumulation_steps: int = 4,
                 use_xbm: bool = True,  # <--- 新增：是否使用XBM
                 xbm_memory_size: int = 16384):  # <--- 新增：XBM记忆库大小

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
                model.final_classifier.parameters()
            )
            
            # 3. 创建参数组列表
            optimizer_parameters = [
                {"params": backbone_params, "lr": learning_rate},
                {"params": head_params, "lr": head_lr}
            ]

            optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate, weight_decay=CONFIG.weight_decay())
        else: # 默认为 Adam
            optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate, weight_decay=CONFIG.weight_decay())

        # --- [XBM修改] 初始化XBM记忆库和损失函数 ---
        self.use_xbm = use_xbm
        if use_xbm:
            # 获取投影头输出维度（需要与模型保持一致）
            proj_config = CONFIG.projection_bridge_config()
            feat_dim = proj_config['hidden_dims'][-1]  # 最后一层的维度，例如256
            
            # 初始化XBM
            self.xbm = XBM(memory_size=xbm_memory_size, feat_dim=feat_dim, device=device)
            
            # 使用支持XBM的损失函数
            self.sup_con_loss = SupConLossWithXBM(temperature=0.1, xbm=self.xbm)
            print(f"[INFO] 已启用XBM，记忆库大小: {xbm_memory_size}")
        else:
            self.xbm = None
            self.sup_con_loss = SupConLoss(temperature=0.1)
            print("[INFO] 未启用XBM，使用标准批内对比")
        
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
    
    # ===== 封装的通用方法 =====
    
    def _perform_optimization_step(self, step: int):
        """
        执行梯度裁剪、优化器步骤、学习率更新等通用优化操作。
        
        Args:
            step (int): 当前训练步数
        
        Returns:
            bool: 是否执行了优化步骤
        """
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # 1. 梯度裁剪
            self.scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # 2. 优化器更新权重
            self.scaler.step(self._optimizer)
            self.scaler.update()
            # 3. 更新学习率
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            # 4. 清空梯度
            self._optimizer.zero_grad()
            return True
        return False
    
    def _cleanup_memory(self, step: int, interval: int = 20):
        """
        定期清理GPU和CPU内存。
        
        Args:
            step (int): 当前步数
            interval (int): 清理间隔
        """
        if step % interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    def _run_validation_epoch(self, val_dataloader, epoch: int) -> tuple:
        """
        运行一个完整的验证epoch，计算损失、准确率和UAR。
        
        Args:
            val_dataloader: 验证数据加载器
            epoch (int): 当前epoch数
        
        Returns:
            tuple: (avg_val_loss, avg_val_acc, val_uar)
        """
        self.model.eval()
        epoch_val_losses, epoch_val_accuracies = [], []
        all_preds, all_labels = [], []
        
        val_loader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch} [验证中]")
        with torch.no_grad():
            for batch in val_loader:
                if not batch: 
                    continue
                
                # 获取logits和真实标签
                logits, labels = self._get_logits_and_real(batch)
                
                # 计算交叉熵损失
                val_loss = self.cross_entropy_loss(logits, labels)
                preds = torch.argmax(logits, dim=1)
                val_accuracy = (preds == labels).float().mean()
                
                # 收集指标
                epoch_val_losses.append(val_loss.item())
                epoch_val_accuracies.append(val_accuracy.item())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均指标
        avg_val_loss = np.mean(epoch_val_losses)
        avg_val_acc = np.mean(epoch_val_accuracies)
        
        # 计算UAR
        from sklearn.metrics import recall_score
        val_uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_val_loss, avg_val_acc, val_uar
    
    def _log_epoch_summary(self, epoch: int, train_loss: float, train_acc: float,
                          val_loss: float = None, val_acc: float = None, val_uar: float = None):
        """
        打印epoch总结日志。

        """
        log_msg = f"Epoch {epoch} 总结: 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}"
        
        if val_loss is not None and val_acc is not None and val_uar is not None:
            log_msg += f", 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}, 验证UAR: {val_uar:.4f}"
        
        logger.info(log_msg)

    def _cleanup_old_checkpoints(self):
        """在训练开始前，清理与当前训练器名称匹配的旧检查点。"""
        save_dir = CONFIG.saved_ckpt_location()
        prefix_to_delete = f"{self._name}_"
        logger.info(f"训练开始前，清理前缀为 '{prefix_to_delete}' 的旧模型...")
        try:
            # 确保目录存在
            if not os.path.exists(save_dir):
                logger.warning(f"检查点目录 {save_dir} 不存在，无需清理。")
                return
            
            for filename in os.listdir(save_dir):
                if filename.startswith(prefix_to_delete):
                    file_path_to_delete = os.path.join(save_dir, filename)
                    os.remove(file_path_to_delete)
                    logger.info(f"已删除旧模型: {file_path_to_delete}")
        except Exception as e:
            logger.error(f"删除旧模型时出错: {e}")
    
    # ===== 原有方法 =====

    def _get_outputs_and_labels(self, batch: dict, use_text_modality: bool = True):
        """
        一个辅助方法，用于处理输入批次并从模型获取所有输出。
        Args:
        use_text_modality (bool): 是否使用文本模态进行门控融合
                                 True=训练模式（双模态融合）
                                 False=验证/推理模式（纯声学）
        """
        # 从批次数据中解包音频、文本和标签
        audio_inputs = batch['audio_input_values'].to(device, non_blocking=True)
        text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
        text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 使用混合精度进行前向传播
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            acoustic_embedding, text_embedding, audio_logits, pooled_fused_features = self.model(
                audio_input_values=audio_inputs,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                use_text_modality=use_text_modality  # 控制是否使用文本模态
            )

        return acoustic_embedding, text_embedding, audio_logits, labels

    # --- *** 新增这个用于评估的方法 *** ---
    def _get_logits_and_real(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为评估阶段实现此方法，以适配 AbstractTrainer 中的 eval 循环。
        该方法遵循“测试时单模态”的原则，只使用声学分支。
        [关键修改]:
        - 设置 use_text_modality=False 来模拟纯声学推理
        - 门控机制会自动将 λ 设为 0，只使用 Ha
        """
        # 1. 从批次中解包音频输入和真实标签
        audio_inputs = batch['audio_input_values'].to(device, non_blocking=True)
        text_input_ids = batch['text_input_ids'].to(device, non_blocking=True)
        text_attention_mask = batch['text_attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # 2. 模型前向传播（只获取 audio_logits）
        # 我们调用模型，但只关心第三个返回值
        _, _, audio_logits, _ = self.model(
            audio_input_values=audio_inputs,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            use_text_modality=False  # <-- 关键修改：不使用文本模态
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
        """
        if not hasattr(self, 'best_uar') or val_uar > self.best_uar:
            self.best_uar = val_uar
            save_path = os.path.join(CONFIG.saved_ckpt_location(), f'{self._name}_model_epoch_{epoch}.pt')
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"新的最佳UAR模型已保存: {val_uar:.4f} (Epoch {epoch}) -> {save_path}")
    
    def _initialize_training(self, train_dataloader) -> transformers.optimization.LambdaLR:
        """
        初始化训练所需的学习率调度器。
        
        Args:
            train_dataloader: 训练数据加载器
        
        Returns:
            学习率调度器
        """
        total_steps = len(train_dataloader) // self.gradient_accumulation_steps * self._num_epochs
        total_steps = max(1, total_steps)
        
        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        return scheduler
    
    def train(self, train_dataloader, val_dataloader=None):
        """
        重写核心的训练方法。
        [关键修改]:
        - 训练时使用 use_text_modality=True（双模态门控融合）
        - 验证时通过 _get_logits_and_real 自动使用 use_text_modality=False（纯声学）
        """
        # --- [重构] 调用封装的方法来清理旧检查点 ---
        self._cleanup_old_checkpoints()
        
        # 初始化学习率调度器
        self.scheduler = self._initialize_training(train_dataloader)

        # 用于存储每个epoch的平均指标
        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []

        logger.info(f"开始训练 {self._name} 模型...")

        for epoch in range(1, self._num_epochs + 1):
    
            # --- 训练循环 (一个 Epoch) ---
            self.model.train()
            epoch_train_losses, epoch_train_accuracies = [], []
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中]")

            for step, batch in enumerate(loader):
                try:
                    # --- 2. 前向传播与损失计算 ---
                    acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(
                        batch, use_text_modality=True)  # 训练时使用双模态融合

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        loss_inter_contrastive = self.sup_con_loss(acoustic_embedding, text_embedding, labels) # 计算监督对比损失
                        loss_ce = self.cross_entropy_loss(audio_logits, labels) # 计算交叉熵损失

                        total_loss = self.alpha * (loss_inter_contrastive) + loss_ce # 根据 alpha 加权组合损失
                        scaled_loss = total_loss / self.gradient_accumulation_steps # 用于梯度累积的损失缩放

                    # --- 3. 反向传播 ---
                    self.scaler.scale(scaled_loss).backward()
                    
                    # --- [XBM修改] 更新记忆库 ---
                    # 必须在反向传播后、优化器更新前进行
                    # 关键：必须detach特征，避免梯度回传到记忆库
                    if self.use_xbm:
                        with torch.no_grad():
                            # 归一化特征（与损失函数内保持一致）
                            acoustic_norm = nn.functional.normalize(acoustic_embedding.detach(), p=2, dim=1)
                            if text_embedding is not None:
                                text_norm = nn.functional.normalize(text_embedding.detach(), p=2, dim=1)
                                # 将双模态特征和标签都入队
                                feats_to_store = torch.cat([acoustic_norm, text_norm], dim=0)
                                labels_to_store = labels.repeat(2)
                            else:
                                feats_to_store = acoustic_norm
                                labels_to_store = labels
                            
                            # 入队新特征，出队旧特征
                            self.xbm.enqueue_dequeue(feats_to_store, labels_to_store)

                    # [梯度检查] 检测并处理异常梯度（改进版）
                    has_nan_grad = False
                    nan_param_count = 0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                # 只清零异常参数的梯度，不影响其他参数
                                param.grad.zero_()
                                has_nan_grad = True
                                nan_param_count += 1
                                # 只在前几次显示详细信息
                                if step < 50:
                                    print(f"⚠️  异常梯度已清零: {name}")

                    if has_nan_grad:
                        # 不再 continue，让正常参数继续更新
                        if step < 50:
                            print(f"   检测到 {nan_param_count} 个参数异常梯度，已清零，继续优化...")

                    # 计算准确率 (用于监控)
                    with torch.no_grad():
                        accuracy = (torch.argmax(audio_logits, dim=1) == labels).float().mean()

                    epoch_train_losses.append(total_loss.item())
                    epoch_train_accuracies.append(accuracy.item())

                    # 梯度累积步骤
                    if self._perform_optimization_step(step):
                        # 更新进度条显示

                        loader.set_postfix(
                            loss=total_loss.item(), 
                            acc=accuracy.item(), 
                            ce=loss_ce.item(), 
                            inter_con=loss_inter_contrastive.item(), # 跨模态对比损失
                        )

                    # 清理内存
                    self._cleanup_memory(step)

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    self._cleanup_memory(step, interval=1)

            # 计算并存储该epoch的平均训练指标
            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))

            # --- 验证循环 (使用封装的方法) ---
            if val_dataloader:
                avg_val_loss, avg_val_acc, val_uar = self._run_validation_epoch(val_dataloader, epoch)
                
                history_val_loss.append(avg_val_loss)
                history_val_acc.append(avg_val_acc)
                
                # 打印epoch总结
                self._log_epoch_summary(epoch, history_train_loss[-1], history_train_acc[-1],
                                       avg_val_loss, avg_val_acc, val_uar)
                
                # 保存最佳UAR模型
                if epoch > 5:
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
        # --- [重构] 调用封装的方法来清理旧检查点 ---
        self._cleanup_old_checkpoints()

        # 初始化学习率调度器
        self.scheduler = self._initialize_training(train_dataloader)

        history_train_loss, history_train_acc = [], []
        history_val_loss, history_val_acc = [], []

        logger.info(f"开始训练 {self._name} 模型...")

        for epoch in range(1, self._num_epochs + 1):
            self.model.train()
            epoch_train_losses, epoch_train_accuracies = [], []
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch} [训练中]")

            for step, batch in enumerate(loader):
                try:
                    acoustic_embedding, text_embedding, audio_logits, labels = self._get_outputs_and_labels(
                        batch, use_text_modality=True)  # 训练时使用双模态融合

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        # 1. 分类损失的计算方式保持不变
                        loss_ce = self.cross_entropy_loss(audio_logits, labels)

                        # 3. [核心修正] 计算无监督对比损失
                        #    我们的新 InfoNCELoss 期望 (view1, view2)
                        
                        # a. 跨模态损失 (Audio-Text)
                        #    直接传入两个视图，不再需要手动创建标签
                        loss_inter_contrastive = self.contrastive_loss(acoustic_embedding, text_embedding)
                                             
                        # 3. 计算总损失，保持公平的结构
                        total_loss = self.alpha * loss_inter_contrastive + loss_ce
                        scaled_loss = total_loss / self.gradient_accumulation_steps

                    self.scaler.scale(scaled_loss).backward()

                    with torch.no_grad():
                        accuracy = (torch.argmax(audio_logits, dim=1) == labels).float().mean()

                    epoch_train_losses.append(total_loss.item())
                    epoch_train_accuracies.append(accuracy.item())

                    if self._perform_optimization_step(step):
                        # 更新进度条显示
                        loader.set_postfix(
                            loss=total_loss.item(), acc=accuracy.item(), ce=loss_ce.item(), 
                            inter_con=loss_inter_contrastive.item()
                        )

                    self._cleanup_memory(step)

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    self._cleanup_memory(step, interval=1)

            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))

            # 验证循环 (使用封装的方法)
            if val_dataloader:
                avg_val_loss, avg_val_acc, val_uar = self._run_validation_epoch(val_dataloader, epoch)
                
                history_val_loss.append(avg_val_loss)
                history_val_acc.append(avg_val_acc)
                
                # 打印epoch总结
                self._log_epoch_summary(epoch, history_train_loss[-1], history_train_acc[-1],
                                       avg_val_loss, avg_val_acc, val_uar)
                
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
        # 从批次中解包音频视图和标签
        audio_inputs = batch['audio_input_values'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 使用混合精度进行前向传播
        # self.model 在这里是 AcousticSupConModel 的一个实例
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            embeddings, audio_logits = self.model(
                audio_input_1=audio_inputs
            )

        return embeddings, audio_logits, labels

    # 步骤 3: 重写验证阶段的"桥梁"方法，以适配纯声学模型
    def _get_logits_and_real(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
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
        _, audio_logits = self.model(
            audio_input_1=audio_inputs
        )

        # 3. 返回 eval 方法期望的两个值
        return audio_logits, labels

    def train(self, train_dataloader, val_dataloader=None):
        """
        [核心修改] 重写核心训练方法，以适配新的损失计算逻辑。
        大部分代码结构与父类 `ContrastiveTrainer.train` 相同。
        """
        # --- [重构] 调用封装的方法来清理旧检查点 ---
        self._cleanup_old_checkpoints()

        # 初始化学习率调度器
        self.scheduler = self._initialize_training(train_dataloader)
        
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
                    embeddings, audio_logits, labels = self._get_outputs_and_labels(batch)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        # 2. 计算损失 [逻辑修改点]
                        #    将两个声学嵌入送入 SupConLoss
                        loss_sup_con = self.sup_con_loss(embeddings, None, labels)

                        #    交叉熵损失计算方式不变
                        loss_ce = self.cross_entropy_loss(audio_logits, labels)

                        #    组合损失公式不变
                        total_loss = self.alpha * loss_sup_con + loss_ce

                        scaled_loss = total_loss / self.gradient_accumulation_steps

                    # 3. 反向传播
                    self.scaler.scale(scaled_loss).backward()

                    with torch.no_grad():
                        accuracy = (torch.argmax(audio_logits, dim=1) == labels).float().mean()

                    epoch_train_losses.append(total_loss.item())
                    epoch_train_accuracies.append(accuracy.item())

                    if self._perform_optimization_step(step):
                        loader.set_postfix(loss=total_loss.item(), acc=accuracy.item(),
                                           sup_con=loss_sup_con.item(), ce=loss_ce.item())

                    self._cleanup_memory(step)

                except Exception as e:
                    tqdm.tqdm.write(f"训练步骤 {step} 出错: {e}")
                    self._cleanup_memory(step, interval=1)

            # 记录训练历史
            history_train_loss.append(np.mean(epoch_train_losses))
            history_train_acc.append(np.mean(epoch_train_accuracies))

            # 验证循环 (使用封装的方法)
            if val_dataloader:
                avg_val_loss, avg_val_acc, val_uar = self._run_validation_epoch(val_dataloader, epoch)
                
                history_val_loss.append(avg_val_loss)
                history_val_acc.append(avg_val_acc)
                
                # 打印epoch总结
                self._log_epoch_summary(epoch, history_train_loss[-1], history_train_acc[-1],
                                       avg_val_loss, avg_val_acc, val_uar)
                
                self._save_best_model_if_improved(val_uar, epoch)

        # 绘图
        self.plot_histories(history_train_loss, history_train_acc, history_val_loss, history_val_acc)