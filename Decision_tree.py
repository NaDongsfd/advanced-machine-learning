import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
import math
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=0, names=col_names)

pima.head()
pima.describe()

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols]  # 特征
y = pima.label  # 标签

# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器对象
#clf = DecisionTreeClassifier()

# 训练决策树分类器
#clf = clf.fit(X_train, y_train)
# 创建决策树分类器对象
clf = DecisionTreeClassifier(max_depth=3)

# 训练决策树分类器
clf = clf.fit(X_train, y_train)

# 预测测试数据的响应
y_pred = clf.predict(X_test)

# 计算模型准确性
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# 预测测试数据的响应
y_pred = clf.predict(X_test)

# 可视化决策树
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))  # 设置绘图尺寸（以英寸为单位）
tree.plot_tree(clf, fontsize=10)
plt.show()

math.exp(-1)