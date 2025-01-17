import seaborn as sns
import matplotlib.pyplot as plt
import MathModels.Plot.styles as styles

plt.style.use(styles.mp_seaborn_light())

# 随机生成数据
data = sns.load_dataset('iris')

# 创建箱形图
plt.figure(figsize=(10, 8))
sns.boxplot(
    x='species',  # 按照 species 分组
    y='sepal_length',  # 以 sepal_length 为变量
    data=data,  # 使用的数据
    palette='Set2'
)
plt.title('鸢尾花萼片长度箱形图', fontsize=16, pad=20)
plt.xlabel('种类', fontsize=12)
plt.ylabel('萼片长度', fontsize=12)
plt.tight_layout()
plt.show()