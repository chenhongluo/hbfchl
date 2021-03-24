# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
import os

from HBF.HBF.nodeAlloc.sort import sort_strings_with_emb_numbers

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

dataDir = "/Users/meituanbijiben/PycharmProjects/hbfhbf/HBF/HBF/nodeAlloc/"
datas = os.listdir(dataDir)
gds = {}
prefix = "nodeAlloc"
gpu ="GTX2080Ti"
graphNames = ["USA-road-d.CAL.gr","delaunay_n20.graph","rmat.3Mv.20Me","ldoor.mtx"]
removeDatas = []
for data in datas:
    names = data.split("..")
    if names[0] == prefix and names[2] in graphNames:
        if names[2] not in gds.keys():
            gds[names[2]] = {}
        gds[names[2]][names[3]] = data
print(gds)
dataSize = len(gds)
kk = 2

def sortedDictValues2(adict):
    keys = adict.keys()
    keys.sort()
    return [dict[key] for key in keys]

def getPlotData(file):
    xs = []
    ys = []
    with open(dataDir + file, 'r') as f:
        level = 0
        line = f.readline()
        while line != '':
            level = level + 1
            if(level <= 7):
                line = f.readline()
                continue
            line = line[0:-1]
            vs = line.split(' ')
            if int(vs[1]) > 0 and int(vs[1]) <200000:
                xs.append(int(vs[1]))
                ys.append(float(vs[2]))
            line = f.readline()
    return xs,ys
canUseStyle = ['bx--','cx-','bo-','r1-','y2-','k3-','g4-','ms-','co-']
ccc = int((dataSize + kk -1)/kk)
fig, ax1 = plt.subplots(figsize=(6 * kk, 6*ccc))
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.5, hspace=0.5)
for i,graphName in zip(range(len(gds.keys())),gds.keys()):
    ds = gds[graphName]
    ax = plt.subplot(int((dataSize + kk -1) / kk), kk, i + 1)
    keys = sort_strings_with_emb_numbers(ds.keys())
    for j,d in zip(range(len(keys)),keys):
        dd = ds[d]
        xs,ys = getPlotData(dd)
        plt.plot(xs, ys,canUseStyle[j],label=d)

    # plt.axis([0, len(xs) + 1, 0, 101])
    plt.xlabel('并行度')
    plt.ylabel('运行时间(ms)')
    # y_major_locator = MultipleLocator(1000)
    # ax.yaxis.set_major_locator(y_major_locator)
    # x_major_locator = MultipleLocator(int(len(xs)/5))
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.title(graphName)
    plt.legend(loc='upper right')

# plt.tight_layout()
plt.savefig("nodeAlloc1.jpg",bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
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
