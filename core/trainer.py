import numpy as np
import tqdm
import logging

import torch
import torch.nn as nn
import transformers
from matplotlib import pyplot as plt
from typing import List

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, recall_score # 增加 recall_score 的导入


from vizualisers.plots import PlotVisualizer

logger = logging.getLogger(__name__)


class AbstractTrainer:
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        optimizer,
        loss: callable,
        name: str,
        alpha: float = None  # <-- 添加了可选的 alpha 参数
    ):
        self.model = model
        self._num_epochs = num_epochs
        self._optimizer = optimizer
        self._loss = loss
        self._name = name
        self._alpha = alpha

    def _get_logits_and_real(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        # 这是一个待实现的抽象方法。
        # 它的任务是从一个数据批次（batch）中，提取出模型的输入数据，并返回模型的原始输出（logits）和真实标签（real）。
        pass

    def train(self, train_dataloader: DataLoader):
        total_steps = len(train_dataloader) * self._num_epochs
        scheduler = transformers.get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        history_loss = []
        history_acc = []

        logger.info(f"Training the {self._name} model...")
        self.model.train()
        for epoch in range(1, self._num_epochs + 1):
            loader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}")
            for batch in loader:
                # Forward
                logits, real = self._get_logits_and_real(batch)
                loss = self._loss(logits, real)

                # Backward
                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self._optimizer.step()
                scheduler.step()

                # Perform the accuracy
                preds = torch.argmax(logits, dim=1)
                accuracy = torch.mean((preds == real).float())

                # Register the metrics
                loss = loss.item()
                accuracy = accuracy.item()
                history_loss.append(loss)
                history_acc.append(accuracy)
                loader.set_postfix(loss=loss, accuracy=accuracy)

        self.plot_histories(history_loss, history_acc)

    # def plot_histories(
    #     self, history_losses: List[float], history_accuracies: List[float]
    # ):
    #     PlotVisualizer.plot_many(
    #         (1, 2),
    #         lambda: PlotVisualizer.plot_history(
    #             history_losses, f"{self._name} loss history"
    #         ),
    #         lambda: PlotVisualizer.plot_history(
    #             history_accuracies, f"{self._name} accuracy history"
    #         ),
    #         filename=f"{self._name}-losses.png",
    #     )

    def plot_histories(
        self,
        train_losses: List[float],
        train_accuracies: List[float],
        val_losses: List[float],
        val_accuracies: List[float]
    ):
        """
        准备数据并调用可视化工具来绘制训练和验证曲线。
        """
        title_suffix = f" (α={self._alpha})" if self._alpha is not None else ""
        filename_suffix = f"-alpha{self._alpha}" if self._alpha is not None else ""

        PlotVisualizer.plot_many(
            (1, 2),
            lambda: PlotVisualizer.plot_history(
                {'Training': train_losses, 'Validation': val_losses},
                f"{self._name} Loss vs. Epochs{title_suffix}"
            ),
            lambda: PlotVisualizer.plot_history(
                {'Training': train_accuracies, 'Validation': val_accuracies},
                f"{self._name} Accuracy vs. Epochs{title_suffix}"
            ),
            filename=f"{self._name}-training-curves{filename_suffix}.png",
        )

    def eval(self, test_dataloader: DataLoader, labels: List[str] = None):
        y_actual = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            loader = tqdm.tqdm(test_dataloader, "Evaluating the model")
            for batch in loader:
                # --- *** 新增的检查 *** ---
                if not batch: # 如果批次为空字典或None，则跳过
                    continue
                # -------------------------
                logits, real = self._get_logits_and_real(batch)
                preds = torch.argmax(logits, dim=1)
                y_actual += real.cpu().numpy().tolist()
                y_pred += preds.cpu().numpy().tolist()

        possible_values = sorted({*y_actual, *y_pred})
        if labels:
            possible_labels = [
                label for i, label in enumerate(labels) if i in possible_values
            ]
        else:
            possible_labels = [str(v) for v in possible_values]
        if len(possible_labels) < len(possible_values):
            possible_labels += ["unknown"] * (
                len(possible_values) - len(possible_labels)
            )
        conf_matrix = confusion_matrix(y_actual, y_pred)

        # --- 新增代码 ---
        # 计算 WAR (Weighted Average Recall, a.k.a. Accuracy)
        war = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

        # 计算 UAR (Unweighted Average Recall, a.k.a. macro average recall)
        uar = recall_score(y_actual, y_pred, average='macro')

        print(f"Accuracy (WAR): {war:.4f}")
        print(f"UAR: {uar:.4f}")
        # --- 结束新增 ---

        # print("Accuracy:", np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))
        filename_suffix = f"-alpha{self._alpha}" if self._alpha is not None else ""

        PlotVisualizer.plot_confusion_matrix(
            conf_matrix, possible_labels, filename=f"{self._name}-conf_matrix{filename_suffix}.png"
        )

        return uar, war, conf_matrix # <--- 返回三个值：UAR, WAR, 混淆矩阵
