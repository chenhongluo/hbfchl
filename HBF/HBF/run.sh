#！/bin/bash
dataDir=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr as-Skitter.mtx asia.osm.graph)
cd /home/chl/HBF/hbf/HBF/HBF/build
mkdir -p /home/chl/hbfrun/

for d in ${dataDir[@]}
do
	echo test ${d}
	./HBF /home/chl/data/${d} ../config_run_v0_32.ini >> /home/chl/hbfrun/run_v0_32.log
done