#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia_osm.mtx)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/nodeTime
datadir=/home/chl/data
gpu=GTX2080Ti
prefix=HBFV1Normal
device=1

d=circuit5M_dc.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 16 normal 41.4 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=delaunay_n20.graph
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 16 normal 16.3 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=flickr.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 normal 19.9 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=ldoor.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 normal 2.3 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=msdoor.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 normal 2.4 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=rmat.3Mv.20Me
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 32 normal 6.6 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=USA-road-d.CAL.gr
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 8 normal 4143.7 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=USA-road-d.USA.gr
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 8 normal 3732.7 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=asia_osm.mtx
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} run V1 8 normal 50.4 100 10 > ${logdir}/nodeTime..cpu..${d}..${prefix}