""" 绘图颜色助手 """

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans

recommend_colormaps = [
    "summer",
    "hsv",
    "Pastel1",
    "Paired",
    "Set2",
    "Set3",
    "tab20c",
    "terrain",
    "gist_rainbow",
    "rainbow",
]


def get_plasma(num_labels):
    """获取 plasma 颜色映射, 等高线图"""
    return ListedColormap(plt.cm.plasma(np.linspace(0, 1, num_labels)))


def get_inferno(num_labels):
    """获取 inferno 颜色映射， 黑热图"""
    return ListedColormap(plt.cm.inferno(np.linspace(0, 1, num_labels)))


def get_cividis(num_labels):
    """获取 cividis 颜色映射， 较好的配色方案"""
    return ListedColormap(plt.cm.cividis(np.linspace(0, 1, num_labels)))


def get_viridis(num_labels):
    """获取 viridis 颜色映射，绿色主导的配色方案"""
    return ListedColormap(plt.cm.viridis(np.linspace(0, 1, num_labels)))


def get_two_colors():
    """获取两种颜色的颜色映射"""
    return ListedColormap(["#ef8a62", "#67a9cf"])


def get_three_colors():
    """获取三种颜色的颜色映射"""
    return ListedColormap(["#DE6E66", "#5096DE", "#CBDE3A"])


def get_four_colors():
    """获取四种颜色的颜色映射"""
    return ListedColormap(["#DE66C2", "#5096DE", "#DEA13A", "#61DE45"])


def monochrome_gradient(num_labels):
    """获取单色渐变颜色映射"""
    colormap = plt.get_cmap("Blues")
    colors = colormap(np.linspace(0, 1, num_labels))
    return ListedColormap(colors)


if __name__ == "__main__":
    # 生成随机数据
    np.random.seed(42)
    data = np.random.rand(100, 2)
    # 创建 KMeans 模型
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(data)
    # 获取颜色映射
    cmap = get_cividis(5)
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, s=100, alpha=0.7)
    plt.title("KMeans", fontsize=16, pad=20)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.tight_layout()
    plt.show()
