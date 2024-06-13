from math import log2

# 计算交叉熵函数
def cross_entropy(p, q):
    return -sum([p[i] * log2(q[i]) for i in range(len(p))])

# 定义数据0.6*4/15
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]

# 计算交叉熵 H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)