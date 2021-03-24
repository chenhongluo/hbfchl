#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia_osm.mtx)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/
datadir=/home/chl/data
gpu=GTX2080Ti

for d in ${datas[@]}
do
	CUDA_VISIBLE_DEVICES=1 ./HBF ${datadir}/${d} none BoostDijstra 1 none 0 100 10 >> nodeInfo.txt
done