import numpy as np

# w1为列向量
x11 = np.array([[-3 / 4, -1 / 4, -1 / 8]]).T
x12 = np.array([[5 / 4, -1 / 4, -1 / 8]]).T
x13 = np.array([[5 / 4, -1 / 4, 7 / 8]]).T
x14 = np.array([[1 / 4, 7 / 4, -1 / 8]]).T
x21 = np.array([[-3 / 4, -1 / 4, 7 / 8]]).T
x22 = np.array([[-3 / 4, 3 / 4, -1 / 8]]).T
x23 = np.array([[-3 / 4, -9 / 4, 7 / 8]]).T
x24 = np.array([[1 / 4, 3 / 4, -17 / 8]]).T

x = [x11, x12, x13, x14, x21, x22, x23, x24]

# 列向量乘行向量
R1 = 0
for i in x:
    R1 += i * i.T;
R = 1 / 8 * R1

# 计算矩阵R的行列式（只要行列式不等于0，就可以求特征值和特征向量）
b = np.linalg.det(R)
# print(b)
# 特征值和特征向量
c = np.linalg.eig(R)
# print(c)
# 特征值
print(c[0])
# 特征向量
print(c[1])