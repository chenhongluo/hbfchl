#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/nodeAlloc
datadir=/home/chl/data
gpu=GTX2080Ti

for d in ${datas[@]}
do
	echo "run: ${d}"
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 1 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW1
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 2 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW2
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 4 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW4
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 8 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW8
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 16 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW16
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 32 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW32
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 64 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW64
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V2 128 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW128
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V3 32 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..DW
done