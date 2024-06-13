import numpy as np

# 给定数据
X = np.array([[7, 4, 3], [4, 1, 8], [6, 3, 5], [8, 6, 1], [8, 5, 7],
              [7, 2, 9], [5, 3, 3], [9, 5, 8], [7, 4, 5], [8, 2, 2]])

# 步骤 1: 数据标准化
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 步骤 2: 计算协方差矩阵
cov_matrix = np.cov(X_centered, rowvar=False)

# 步骤 3: 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 步骤 4: 选择前两个主要成分
sorted_indices = np.argsort(eigenvalues)[::-1]
top2_eigenvectors = eigenvectors[:, sorted_indices[:2]]

# 步骤 5: 降维
X_reduced = np.dot(X_centered, top2_eigenvectors)

# 步骤 6: 恢复数据到原始维度
X_recovered = np.dot(X_reduced, top2_eigenvectors.T) + X_mean

# 步骤 7: 计算均方误差（MSE）
mse = np.mean((X - X_recovered) ** 2)

print("Original Data:\n", X)
print("Recovered Data:\n", X_recovered)
print("Mean Squared Error (MSE):", mse)