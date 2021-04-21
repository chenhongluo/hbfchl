#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia_osm.mtx)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/nodeValid
datadir=/home/chl/data
gpu=GTX2080Tinorm
device=7

d=circuit5M_dc.mtx
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 16 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 16 ${prefix} 296.6 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 16 ${prefix} 25 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=delaunay_n20.graph
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 269.6 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 16 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=ldoor.mtx
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 40.5 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 2.3 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=msdoor.mtx
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 50.5 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 2.2 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=rmat.3Mv.20Me
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 60.5 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 6.6 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/rmat.3Mv.ddsg cacValid V1 32 normal 3.5 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..rmat.3Mv.20Me..none
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/flickr.ddsg cacValid V1 32 normal 400 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..flickr.ddsg..BCE
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/flickr.ddsg cacValid V1 32 delta 6000 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..flickr.ddsg..delta
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/flickr.ddsg cacValid V1 32 none 400 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..flickr.ddsg..none

CUDA_VISIBLE_DEVICES=7 ./HBF /home/chl/data/flickr.ddsg cacValid V1 32 normal 400 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..flickr.ddsg..BCE


d=USA-road-d.CAL.gr
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 1500000 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 3300 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=USA-road-d.USA.gr
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 800000 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 2300 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}

d=asia_osm.mtx
prefix=none
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 41.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=delta
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 80000 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
prefix=normal
CUDA_VISIBLE_DEVICES=${device} ./HBF ${datadir}/${d} cacValid V1 32 ${prefix} 50.4 100 1 0 > ${logdir}/nodeTime..cpu..${d}..${prefix}
