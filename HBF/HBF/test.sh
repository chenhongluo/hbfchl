#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr as-Skitter.mtx asia.osm.graph)
configs=(test_v0_32 test_v0_16 test_v1_32 test_v1_16 run_v0_32 run_v0_16)
cd /home/chl/HBF/hbf/HBF/HBF/build
git pull && make
mkdir -p /home/chl/hbflogs/

for c in ${configs[@]}
do
	echo "" > /home/chl/hbflogs/${c}.log
	for d in ${datas[@]}
	do
		echo test ${d}
		./HBF /home/chl/data/${d} /home/chl/hbfconfigs/config_${c}.ini >> /home/chl/hbflogs/${c}.log
	done
done
