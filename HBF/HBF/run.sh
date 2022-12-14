#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia.osm.graph)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/nodeAlloc
datadir=/home/chl/data
gpu=GTX2080Ti

for d in ${datas[@]}
do
	echo "run: ${d}"
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V1 1 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW1
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V1 2 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW2
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V1 4 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW4
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V1 8 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW8
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V1 16 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW16
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V1 32 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..VW32
done

	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} nodeAllocTest V3 32 none 0 0 > ${logdir}/nodeAlloc..${gpu}..${d}..DW
