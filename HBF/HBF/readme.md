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

cd /home/chl/hbftest/HBF/HBF/build
git pull && make
git pull && cmake -DCMAKE_BUILD_TYPE=Debug -DARCH=75 .. && make
git pull && cmake -DCMAKE_BUILD_TYPE=Release -DARCH=75 .. && make

CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/flickr.mtx ../config.ini
cuda-memcheck ./HBF /home/chl/data/circuit5M_dc.mtx ../config.ini
bash /home/chl/hbfrun/run.sh
docker run -it --gpus all -v /home/chl:/home/chl -p 10106:22 --network="host" --name hbf2 --rm --privileged 
apt-get update && apt-get install -y openssh-server
passwd
vi /etc/ssh/sshd_config
/etc/init.d/ssh restart
docker run --name=chlmysql -d mysql/mysql-server
docker exec -it chlmysql mysql -uroot -p 
@M3hUNz@dihf3zij3L3vaxFiSeq4 
ALTER USER 'root'@'localhost' IDENTIFIED BY '1234';
ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY '1234';
flush privileges;

确认是release版本
cmake -DCMAKE_BUILD_TYPE=Release -DARCH=75 .. && make
cmake -DCMAKE_BUILD_TYPE=Debug -DARCH=75 .. && make
脚本测:
run.sh 执行
用pycharm拉到本地，然后执行plot.py

target/linux-desktop-glibc_2_11_3-x64/ncu --kernel-id :::100 -f --export /home/chl/nvprof/hbf8 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Deprecated --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/delaunay_n20.graph run V1 8 none 100 1 1 0

target/linux-desktop-glibc_2_11_3-x64/ncu --kernel-id :::1000 --metric dram__sectors_read.sum,dram__sectors_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,lts__t_sectors_op_write.sum /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/delaunay_n24.mtx run V1 32 none 100 1 1 0
./HBF /home/chl/data/flickr.mtx run V1 32 none 100 1 1 0
./HBF /home/chl/data/flickr.mtx run V1 32 none 100 1 1 0

target/linux-desktop-glibc_2_11_3-x64/ncu --metric dram__sectors_read.sum /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/delaunay_n20.graph run V1 4 none 100 1 1 0

datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia_osm.mtx)

target/linux-desktop-glibc_2_11_3-x64/ncu -f --export /home/chl/nvprof/hbf8 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Deprecated --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/avg_graph/avg_100_16.gr  nodeAllocTest V2 8 none 100 1 1 0

target/linux-desktop-glibc_2_11_3-x64/ncu --kernel-id :::0 -f --export /home/chl/nvprof/hbf16 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Deprecated --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/avg_graph/avg_100_16.gr nodeAllocTest V2 16 none 100 1 1 0

target/linux-desktop-glibc_2_11_3-x64/ncu --kernel-id :::0 -f --export /home/chl/nvprof/hbf8 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Deprecated --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/avg_graph/avg_100_16.gr nodeAllocTest V2 8 none 100 1 1 0

target/linux-desktop-glibc_2_11_3-x64/ncu --metric dram__sectors_read.sum,dram__sectors_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_atom.sum,	smsp__inst_executed_op_global_ld.sum /home/chl/hbftest/HBF/HBF/build/HBF /home/chl/data/avg_graph/avg_100_16.gr nodeAllocTest V2 8 none 100 1 1 0

CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/rmat.3Mv.20Me nodeAllocTest V2 1 none 0 0 2048 50000

单独测：
CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/asia_osm.mtx cacValid V1 4 none 0 1 1000000

CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/asia_osm.mtx cacValid V3 4 none 0 1 1000

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/data/circuit5M_dc.mtx run V1 16 delta 269.4 100

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/data/circuit5M_dc.mtx run V1 16 normal 15 100

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/data/flickr.mtx test V2 16 normal 15 100

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/ppopp-code/inputs/graph-int/USA-road-d.USA.gr run V2 16 normal 3500 2

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/ppopp-code/inputs/graph-int/USA-road-d.USA.gr test V2 32 normal 2500 100

cd /home/chl/ppopp-code/ &&
python3 verify.py ads_int_final_dist/ nf_int_final_dist/ chl_list_int

cd /home/chl/ppopp-code/nf_int/build/lonestar/analytics/gpu/sssp && make &&
./sssp-gpu -s 0 -o /home/chl/ppopp-code/nf_int_final_dist/USA-road-d.USA.gr /home/chl/ppopp-code/inputs/graph-int/USA-road-d.USA.gr
./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/kron_g500-logn20.gr

cd /home/chl/ppopp-code/ads_int/ &&
./sssp -s 0 -o /home/chl/ppopp-code/ads_int_final_dist/USA-road-d.USA.gr /home/chl/ppopp-code/inputs/graph-int/USA-road-d.USA.gr

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/data/ldoor.mtx run V1 32 delta 32.5 100

CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/rmat.3Mv.20Me cacValid V1 32 normal 60 100 1 1 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..rmat.3Mv.20Me..BCE

CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/rmat.3Mv.20Me run V1 32 delta 60 100 1 0

CUDA_VISIBLE_DEVICES=2 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 38757.9 100


1.5 47979.4
1.9 40171.4
2.2 36247.6
2.6 33638.2
3.0 31882.4
3.3 30345.2 
3.7 29235.0 
4.1 28323.2
4.8 27031.4

CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 40000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 80000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 120000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 160000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 200000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 240000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 280000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 320000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 400000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 500000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 600000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 700000 5 >> chl.txt
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.gr run V1 32 delta 1000000 5 >> chl.txt