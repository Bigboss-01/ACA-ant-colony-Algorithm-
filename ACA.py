import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import random
import copy
# 第一步导入数据
dataFile = "citys_data.mat"
df = scio.loadmat(dataFile)

city = (df["citys"]).astype(int)    # 是一个二维的x,y
n = len(city)   # 城市数
D = np.zeros((n, n))  # 用一个矩阵表示距离
# print(city)
for i in range(n):
    for j in range(n):
        if i == j:
            D[i, j] = 1e-4
        else:
            D[i, j] = np.sqrt((city[j, 0]-city[i, 0])*(city[j, 0]-city[i, 0]) +
                              (city[j, 1]-city[i, 1])*(city[j, 1]-city[i, 1]))


# 初始化参数
m = 50  # 蚂蚁数量
alpha = 1   # 信息素重要程度因子
beta = 5    # 启发函数重要程度因子
rho = 0.1   # 信息素挥发因子
Q = 1   # 常系数
Eta = np.ones((n, n))    # 启发函数
Tau = np.ones((n, n))   # 信息素矩阵
Table = np.zeros((m, n), dtype=np.int)    # 路径记录表
iter = 0    # 迭代次数初值
iter_max = 200  # 最大迭代次数
Route_best = np.zeros((iter_max, n))    # 每一次迭代最好的路径
Length_best = np.zeros((iter_max, 1))    # 每次迭代最佳路径的长度
Length_ave = np.zeros((iter_max, 1))    # 每次迭代平均路径的长度


for i in range(n):
    for j in range(n):
        Eta[i, j] = 1/D[i, j]

# 迭代找出最佳路径
while iter < iter_max:
    # 随机产生各个蚂蚁的起点城市
    start = np.zeros((m, 1), dtype=np.int)
    for i in range(m):
        start[i] = random.randint(0, n-1)
    Table[:, 0] = start[:, 0]
    citys_index = range(0, n)
    for i in range(m):  # 每只蚂蚁的路径
        tabu = []
        for j in range(1, n):
            tabu.append(Table[i, j-1])
            allow = [a for a in citys_index if a not in tabu]
            P = copy.deepcopy(allow)
            # 计算城市间转移概率
            for k in range(len(allow)):
                P[k] = ((Tau[int(tabu[-1]), allow[k]])**alpha)*((Eta[int(tabu[-1]), allow[k]])**beta)
            sum = np.sum(P)
            P = P/sum
            Pc = []
            s2 = 0
            for k in P:
                s2 = s2 + k
                Pc.append(s2)
            # 在[0，1]区间内产生一个均匀分布的伪随机数r；
            # 若r<q[1]，则选择个体1，否则，选择个体k，使得：q[k-1]<r≤q[k] 成立；
            r = random.random()
            target_index = [a for a in Pc if a > r][0]
            target_index = Pc.index(target_index)
            target = allow[target_index]
            Table[i, j] = int(target)
    # 计算每个蚂蚁的路径距离
    L = [0]
    Length = L * m
    for i in range(m):
        Route = Table[i, :]
        for j in range(n-1):
            Length[i] = Length[i] + D[Route[j], Route[j+1]]
        Length[i] = Length[i] + D[Route[n-1], Route[0]]

    # 计算最短路径距离及平均距离
    min_Length = np.min(Length)
    min_index = Length.index(min_Length)
    if iter == 0:
        Length_best[iter] = min_Length
        Route_best[iter, :] = Table[min_index, :]
        Length_ave[iter] = np.mean(Length)
    elif iter >= 1:
        if min_Length-Length_best[iter-1][0] >= 0:
            Length_best[iter] = Length_best[iter-1][0]
        else:
            Length_best[iter] = min_Length
        Length_ave[iter] = np.mean(Length)
        if Length_best[iter] == min_Length:
            Route_best[iter, :] = Table[min_index, :]
        else:
            Route_best[iter, :] = Route_best[iter, :]

    # 更新信息素
    Delta_Tau = np.zeros((n, n))
    # 逐个蚂蚁计算
    for i in range(m):
        # 逐个城市计算
        for j in range(n-1):
            Delta_Tau[Table[i, j], Table[i, j+1]] = Delta_Tau[Table[i, j], Table[i, j+1]] + Q/Length[i]
        Delta_Tau[Table[i, n-1], Table[i, 0]] = Delta_Tau[Table[i, n-1], Table[i, 0]] + Q / Length[i]
    Tau = (1 - rho) * Tau + Delta_Tau
    iter += 1
    Table = np.zeros((m, n), dtype=np.int)
# 结果显示
Shortest_Length = np.min(Length_best)
index = Length_best.tolist().index(Shortest_Length)
Route_temp = Route_best[index, :]
Shortest_Route = [ int(i) for i in Route_temp ]
print("最短距离:", Shortest_Length)
print("最短路径:", Shortest_Route)


# 绘图显示
fig, ax = plt.subplots()
final_route = np.zeros((n, 2), dtype=np.int)
for i in range(len(Shortest_Route)):
    final_route[i, :] = (city[Shortest_Route[i], :])

ax.plot(final_route[:, 0], final_route[:, 1], "r")
for i,txt in enumerate(Shortest_Route):
    if i == 0:
        ax.annotate(("begin", txt+1), (final_route[i, 0], final_route[i, 1]))
    elif i == n-1:
        ax.annotate(("end", txt + 1), (final_route[i, 0], final_route[i, 1]))
    else:
        ax.annotate((txt + 1), (final_route[i, 0], final_route[i, 1]))
plt.show()

plt.plot(Length_best, "r*")
plt.plot(Length_ave, "b-")
plt.show()
