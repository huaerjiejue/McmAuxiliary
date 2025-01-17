import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import warnings
import MathModels.Plot.styles as styles

plt.style.use(styles.mp_seaborn_light())
warnings.filterwarnings('ignore')

penguins = sns.load_dataset("penguins")
# sns.pairplot(penguins, hue="species", diag_kind="hist")
# sns.pairplot(penguins, hue="species", diag_kind="kde")
# sns.pairplot(penguins, hue="species")
"""
diag_kind : {'auto', 'hist', 'kde', None}
    Kind of plot for the diagonal subplots. If 'auto', choose based on whether or not hue is used.
    If None, no plot is drawn.  # 对角线图的类型
hue : name of variable in ``data``
    Variable in ``data`` to map plot aspects to different colors.  # 颜色映射的变量
palette : palette name, list, or dict
    Colors to use for the different levels of the ``hue`` variable.  # 调色板
corner : bool
    If True, don't add axes to the upper (off-diagonal) triangle of the grid, making this a "corner" plot.  # 是否只画一半
g.map_lower(sns.kdeplot, levels=4, color=".2") # 画密度图
"""
g = sns.pairplot(penguins, diag_kind="kde", hue="species", palette="Set2", corner=True)
g.map_lower(sns.kdeplot, levels=4, color=".2")
plt.show()