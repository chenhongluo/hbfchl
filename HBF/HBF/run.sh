#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr as-Skitter.mtx asia.osm.graph)
configs=(profile_v0_32 run_v0_32)
cd /home/chl/HBF/hbf/HBF/HBF/build
git pull && make
configdir=/home/chl/hbfrun/configs
logdir=/home/chl/hbfrun/logs
datadir=/home/chl/data
mkdir -p ${configdir}
mkdir -p ${logdir}

for c in ${configs[@]}
do
	echo "" > ${logdir}/${c}.log
	for d in ${datas[@]}
	do
		echo test ${d}
		./HBF ${datadir}/${d} ${configdir}/config_${c}.ini >> ${logdir}/${c}.log
	done
done
