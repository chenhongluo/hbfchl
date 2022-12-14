# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import math
import os

from HBF.HBF.nodeAlloc.sort import sort_strings_with_emb_numbers

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

dataDir = "/Users/meituanbijiben/PycharmProjects/hbfhbf/HBF/HBF/nodeValid/"
datas = os.listdir(dataDir)
gds = {}
prefix = "nodeValid"
gpu ="GTX2080Ti"
# graphNames = ["USA-road-d.CAL.gr","delaunay_n20.graph","rmat.3Mv.20Me",'asia_osm.mtx','circuit5M_dc.mtx']
graphNames = ["USA-road-d.CAL.gr",'USA-road-d.USA.gr','asia_osm.mtx',
              "delaunay_n20.graph",'msdoor.mtx','ldoor.mtx',
              "rmat.3Mv.20Me",'flickr.mtx','circuit5M_dc.mtx']
removeDatas = []
for data in datas:
    names = data.split("..")
    if names[0] == prefix and names[2] in graphNames:
        gds[names[2]] = data
print(gds)
dataSize = len(gds)
kk = 2

def sortedDictValues2(adict):
    keys = adict.keys()
    keys.sort()
    return [dict[key] for key in keys]

def getPlotData1(file,interval):
    xs = []
    ys = []
    zs = []
    yys = []
    zzs = []
    with open(dataDir + file, 'r') as f:
        line = f.readline()
        while line != '':
            line = line[0:-1]
            if line.startswith('realDepth'):
                vs = line.split(' ')
                interval = int(int(vs[1]) / interval)
                if interval == 0:
                    interval = 1
            if line.startswith("validInfo:") :
                vs = line.split(' ')
                if int(vs[1]) % interval == 0:
                    xs.append(int(vs[1]))
                    ys.append(int(vs[6]))
                    zs.append(int(vs[6]) + int(vs[7]))
                    yys.append(int(vs[8]))
                    zzs.append(int(vs[8]) +int(vs[9]))
            line = f.readline()
    return xs,ys,zs,yys,zzs

def getPlotData2(file):
    levelMap = {}
    with open(dataDir + file, 'r') as f:
        line0 = f.readline()[0:-1]
        line = f.readline()
        while line != '':
            line = line[0:-1]
            if not line.startswith("validInfo:") and line0.startswith("validInfo:"):
                infos = line0.split(' ')
                xs = []
                ys = []
                zs = []
                ts = []
                v = 0
                av = 0
                while 1:
                    av = av+1
                    ws = line.split(' ')
                    xs.append(int(ws[0]))
                    if int(ws[1]) == 1:
                        v = v + 1
                    ys.append(v)
                    zs.append(av)
                    ts.append(v / av * 100)
                    line = f.readline()
                    if line == '':
                        break
                    line = line[0:-1]
                    if line.startswith("validInfo:"):
                        break
                levelMap[int(infos[1])] = {'xs': xs, 'ys': ys, 'zs': zs ,'ts' : ts}
            line0 = line
            line = f.readline()
    return levelMap
canUseStyle = ['bx--','cx-','bo-','r1-','y2-','k3-','g4-','ms-','co-']

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
#                 wspace=0.5, hspace=0.5)
def plot1():
    ccc = int((dataSize + kk - 1) / kk)
    fig, ax1 = plt.subplots(figsize=(5 * kk, 5 * ccc))
    for i,graphName in zip(range(len(gds.keys())),gds.keys()):
        ds = gds[graphName]
        ax = plt.subplot(int((dataSize + kk -1) / kk), kk, i + 1)

        xs,ys,zs,yys,zzs = getPlotData1(ds,10000)
        # plt.plot(xs, ys,canUseStyle[5],label='????????????')
        plt.plot(xs, yys, canUseStyle[5], label='??????????????????')
        # plt.plot(xs, zs, canUseStyle[1], label='????????????')

        # plt.axis([0, len(xs) + 1, 0, 101])
        plt.xlabel('????????????')
        plt.ylabel('????????????')
        # y_major_locator = MultipleLocator(1000)
        # ax.yaxis.set_major_locator(y_major_locator)
        # x_major_locator = MultipleLocator(int(len(xs)/5))
        # ax.xaxis.set_major_locator(x_major_locator)
        plt.title(graphName)
        plt.legend(loc='upper right')
    # plt.tight_layout()
    plt.savefig("valid1.jpg",bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.show()

def plot2():
    kk = 3
    fig, ax1 = plt.subplots(figsize=(6 * kk, 6 * dataSize))
    for i,graphName in zip(range(len(gds.keys())),gds.keys()):
        ds = gds[graphName]
        datas = getPlotData2(ds)
        for j,key in zip(range(len(datas.keys())),datas.keys()):
            if j >= kk:
                break
            ax = plt.subplot(dataSize, kk, i * kk + j + 1)
            ax2 = ax.twinx()
            data = datas[key]
            # plt.plot(xs, ys,canUseStyle[0],label='????????????')
            # plt.plot(data.get('xs'), data.get('zs'), canUseStyle[0], label='??????????????????')
            # plt.plot(data.get('xs'), data.get('yys'), canUseStyle[1], label='??????????????????')
            lns1 = ax.plot(data.get('xs'),data.get('ts') , canUseStyle[5], label='??????????????????')
            lns2 = ax2.plot(data.get('xs'),data.get('ys') , canUseStyle[0], label='??????????????????')
            # interval = (data.get('ys')[len(data.get('ys'))-1] - data.get('ys')[0])/(data.get('xs')[len(data.get('xs'))-1] - data.get('xs')[0])
            # xdata0 = data.get('xs')[0]
            # ydata0 = data.get('ys')[0]
            # xdata = data.get('xs')
            # yyy = []
            # for ttt in range(len(xdata)):
            #     yyy.append(ydata0 + interval * (xdata[ttt] - xdata0))
            # lns3 = ax2.plot(xdata,yyy, canUseStyle[1], label='????????????????????????')
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='upper left')

            # plt.axis([0, len(xs) + 1, 0, 101])
            plt.xlabel('????????????')
            ax2.set_ylabel('??????')
            ax.set_ylabel('??????(%)')
            plt.title(graphName+'???-'+str(key))
        # y_major_locator = MultipleLocator(1000)
        # ax.yaxis.set_major_locator(y_major_locator)
        # x_major_locator = MultipleLocator(int(len(xs)/5))
        # ax.xaxis.set_major_locator(x_major_locator)
    plt.tight_layout()
    plt.savefig("valid2.jpg",bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.show()

plot1()

'''
color?????????????????????r???????????????red???
marker?????????????????????o???????????????????????????circle marker???
linestyle????????????????????????dashed???????????????????????????
'''
# plt.plot(x, y, color='r',marker='o',linestyle='dashed')
#plt.plot(x, y, 'ro')
'''
axis??????????????????
?????????axis[xmin, xmax, ymin, ymax]???
?????????axis[x????????????, x????????????, y????????????, y????????????]
'''
# plt.axis([0, 800000, 0, 0.26])
# y_major_locator=MultipleLocator(0.01)
# #???y???????????????????????????10?????????????????????
# ax=plt.gca()
# #ax???????????????????????????
# ax.yaxis.set_major_locator(y_major_locator)
