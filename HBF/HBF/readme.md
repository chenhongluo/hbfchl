[action]
actions：程序可以执行的动作
    1. run: 执行./HBF
    2. test: 用来验证程序是否正确，结果将于Dijkastra结果比对
    3. predeal: 用来预处理输入的图文件

subaction: action附属
    run的subaction:
        使用cuda还是host
    test的subaction:
        使用cuda还是host，基准对比是自己实现的Dijkstra
    predeal的subaction:
        1. none,图被转义成自定义的chl格式，后缀是.gc
        2. reAssign,节点重编号后转成chl格式，后缀是.re.gc
        3. preCompute,提前计算部分最短路后转成chl格式，后缀是.pc.gc
        
dataDir: 用来存储.gc的文件
testNodeSize：test和run的随机点的数量

[cuda]
gpu: 使用的gpu编号
gridDim
blockDim
sharedLimit：<<<gridDim,blockDim,sharedLimit>>>>
kernel: 使用的kernel编号
atomic64：是否使用原子64位优化
factor=10 #TODO
vwSize=32：vwSize

[host]
random  
seed
random_min
random_max：图没有边长，自动生成的参数
edgeType=0
kernel: 使用的kernel函数

附加：
指标采集，使用cout作为采集接口
python脚本run.py会取得这部分字符串之后处理
默认格式是
*其他输出*
*indicator start output：*
*指标输出*

指标格式是：
key: value

key 有这些参数：
avgTime: 平均执行时间
testNodes: 总共有多个测试点
avgRelaxNodes：平均Relax多少个节点
avgRelaxEdges: 平均Relax多少个边

git pull && make
cmake -DCMAKE_BUILD_TYPE=Debug -DARCH=75 .. && make
cmake -DCMAKE_BUILD_TYPE=Release -DARCH=75 .. && make

./HBF /home/chl/data/flickr.mtx ../config.ini
bash /home/chl/hbfrun/run.sh