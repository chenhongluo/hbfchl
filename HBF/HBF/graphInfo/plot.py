# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

dataDir = "/Users/meituanbijiben/PycharmProjects/hbfhbf/HBF/HBF/graphInfo/"
datas = os.listdir(dataDir)
ds = []
removeDatas = []
for data in datas:
    if data.startswith("graphInfo") and data not in removeDatas:
        ds.append(data)
datas = ds
print(datas)
dataSize = len(datas)
kk = 3

def getPlotData(file):
    xs = []
    ys = []
    with open(dataDir + file, 'r') as f:
        level = 0
        line = f.readline()
        while line != '':
            level = level + 1
            if(level <= 10):
                line = f.readline()
                continue
            line = line[0:-1]
            vs = line.split(' ')
            xs.append(int(vs[0]))
            ys.append(float(vs[4]) * 100)
            line = f.readline()
    return xs,ys

fig, axes = plt.subplots(3, 3, figsize=(6,6))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.5, hspace=0.5)
for i,graph in zip(range(len(datas)),datas):
    s = graph.find(".") + 1
    e = graph.rfind(".")
    graphName = graph[s:e]
    print(graph,graphName)
    ax = plt.subplot(int((dataSize + 1) / kk), kk, i + 1)
    xs,ys = getPlotData(graph)
    plt.plot(xs, ys)
    # plt.axis([0, len(xs) + 1, 0, 101])
    plt.xlabel('节点出度')
    plt.ylabel('占比(%)')
    y_major_locator = MultipleLocator(20)
    ax.yaxis.set_major_locator(y_major_locator)
    x_major_locator = MultipleLocator(int(len(xs)/5))
    ax.xaxis.set_major_locator(x_major_locator)
    plt.title(graphName)

plt.tight_layout()
plt.savefig("degreeDis.jpg")
plt.show()
'''
color：线条颜色，值r表示红色（red）
marker：点的形状，值o表示点为圆圈标记（circle marker）
linestyle：线条的形状，值dashed表示用虚线连接各点
'''
# plt.plot(x, y, color='r',marker='o',linestyle='dashed')
#plt.plot(x, y, 'ro')
'''
axis：坐标轴范围
语法为axis[xmin, xmax, ymin, ymax]，
也就是axis[x轴最小值, x轴最大值, y轴最小值, y轴最大值]
'''
# plt.axis([0, 800000, 0, 0.26])
# y_major_locator=MultipleLocator(0.01)
# #把y轴的刻度间隔设置为10，并存在变量里
# ax=plt.gca()
# #ax为两条坐标轴的实例
# ax.yaxis.set_major_locator(y_major_locator)
