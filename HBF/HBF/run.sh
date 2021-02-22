#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia.osm.graph)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF
datadir=/home/chl/data

for d in ${datas[@]}
do
	echo "run: ${d}"
	./HBF ${datadir}/${d} 100 > ${logdir}/graphInfo.${d}.txt
done