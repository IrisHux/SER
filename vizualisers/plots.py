import os.path
import logging
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from typing import List, Dict

from core.config import CONFIG


class PlotVisualizer:
    @classmethod
    # def plot_history(cls, history: List[float], title: str):
    #     plt.plot(history)
    #     plt.title(title)
    def plot_history(cls, history_dict: Dict[str, List[float]], title: str):
        """
        在同一张图上绘制训练和验证历史曲线。
        参数:
            history_dict (dict): 一个字典，键是标签（如 'Training', 'Validation'），
                                 值是指标值的列表。
            title (str): 图表的标题。
        """
        for label, history in history_dict.items():
            if history:  # 只有当历史列表不为空时才绘制
                plt.plot(history, label=label)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()  # 显示图例以区分曲线
        plt.grid(True) # 添加网格线

    @classmethod
    # def plot_many(cls, dims: (int, int), *args, filename: str = None):
    #     assert dims[0] * dims[1] == len(args)
    #     for idx, arg in enumerate(args, start=1):
    #         plt.subplot(*dims, idx)
    #         arg()
    def plot_many(cls, dims: tuple, *args, filename: str = None):
        """
        在一个Figure中绘制多个子图。
        """
        plt.figure(figsize=(dims[1] * 7, dims[0] * 5))  # 调整图表大小以获得更好的可读性
        assert dims[0] * dims[1] == len(args)
        for idx, plot_func in enumerate(args, start=1):
            plt.subplot(*dims, idx)
            plot_func()
        
        plt.tight_layout()  # 自动调整子图布局

        if filename is None:
            plt.show()
        else:
            plt.savefig(os.path.join(CONFIG.save_plots_location(), filename))
        plt.close()

    @classmethod
    def plot_confusion_matrix(
        cls, confusion_matrix: List[List[int]], labels: List[str], filename: str = None
    ):
        ConfusionMatrixDisplay(confusion_matrix, display_labels=labels).plot()
        if filename is None:
            plt.show()
        else:
            plt.savefig(os.path.join(CONFIG.save_plots_location(), filename))
        plt.close()
