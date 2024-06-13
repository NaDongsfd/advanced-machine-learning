import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 具体的二维数据
X = np.array([[2, 3], [3, 3], [3, 2], [5, 8], [6, 8], [6, 9], [1, 0], [0, 1], [1, 1], [7, 7]])
y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 创建线性核的SVM模型
svc_model = svm.SVC(kernel='linear')

# 训练模型
svc_model.fit(X_train, y_train)

# 预测测试集
y_pred = svc_model.predict(X_test)

# 评估模型
accuracy = svc_model.score(X_test, y_test)
print(f"模型在测试集上的准确率: {accuracy}")

# 输出混淆矩阵
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(conf_matrix)

# 获取超平面的系数和截距
w = svc_model.coef_[0]
b = svc_model.intercept_[0]
print(f"分离超平面的方程: {w[0]}*x1 + {w[1]}*x2 + {b} = 0")

# 可视化分类结果
# 创建一个网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

# 使用模型预测网格中的每个点
Z = svc_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# 绘制训练数据点
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', marker='o', label='Class 0 (train)')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', marker='x', label='Class 1 (train)')

# 绘制测试数据点
plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='red', marker='o', edgecolors='k', label='Class 0 (test)')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='blue', marker='x', edgecolors='k', label='Class 1 (test)')

# 绘制分离超平面
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2 = - (w[0] * x1 + b) / w[1]
plt.plot(x1, x2, color='black', linestyle='--', label='Decision Boundary')

# 设置图形标题和标签
plt.title('SVM 2D Classification')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()