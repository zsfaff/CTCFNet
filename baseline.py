import math

# 输入数据
data1 = [2.8, 2.7, 2.6, 2.58]
data2 = [0.89, 1.1]
data3 = [0.7, 0.8]

# 计算每个数据组内部的权重
def calc_inner_weight(data):
    p = [x/sum(data) for x in data] # 计算每个数据出现的概率
    entropy = -sum([pi*math.log(pi) for pi in p if pi>0]) # 计算熵
    return [entropy/len(data) for _ in data] # 每个数据的权重是熵除以数据数

w1 = calc_inner_weight(data1)
w2 = calc_inner_weight(data2)
w3 = calc_inner_weight(data3)

print("data1内部权重:", w1)
print("data2内部权重:", w2)
print("data3内部权重:", w3)

# 计算三组数据之间的权重
all_data = data1 + data2 + data3
p = [sum(data1), sum(data2), sum(data3)]
p = [x/sum(all_data) for x in p]
entropy = -sum([pi*math.log(pi) for pi in p if pi>0])
w = [entropy/3, entropy/3, entropy/3]

print("三组数据之间的权重:", w)