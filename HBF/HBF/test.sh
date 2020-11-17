#！/bin/bash
dataDir=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr as-Skitter.mtx asia.osm.graph)
cd /home/chl/HBF/hbf/HBF/HBF/build
mkdir -p /home/chl/hbftest/
for d in ${dataDir[@]}
do
	echo test ${d}
	./HBF /home/chl/data/${d} ../config_test_v0_32.ini >> /home/chl/hbftest/test_v0_32.log
done

for d in ${dataDir[@]}
do
	echo test ${d}
	./HBF /home/chl/data/${d} ../config_test_v0_16.ini >> /home/chl/hbftest/test_v0_16.log
done