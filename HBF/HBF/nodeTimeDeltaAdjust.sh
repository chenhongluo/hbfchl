#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia_osm.mtx)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/nodeTime
datadir=/home/chl/data
gpu=GTX2080Ti
prefix=HBFV1Delta
device=2

d=circuit5M_dc.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 16 delta 296.6 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=delaunay_n20.graph
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 16 delta 269.4 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=flickr.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 delta 134.9 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=ldoor.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 delta 32.4 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=msdoor.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 delta 32.5 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=rmat.3Mv.20Me
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 delta 237.6 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=USA-road-d.CAL.gr
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 8 delta 35009.4 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=USA-road-d.USA.gr
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 8 delta 38757.9 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}
13023003401
d=asia_osm.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 8 delta 759.7 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}