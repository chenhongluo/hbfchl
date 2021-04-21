#！/bin/bash
datas=(circuit5M_dc.mtx delaunay_n20.graph flickr.mtx ldoor.mtx msdoor.mtx rmat.3Mv.20Me USA-road-d.CAL.gr USA-road-d.USA.gr asia_osm.mtx)
cd /home/chl/hbftest/HBF/HBF/build
logdir=/home/chl/hbftest/HBF/HBF/nodeTime
datadir=/home/chl/data
gpu=GTX2080Ti
prefix=HBFV0
device=1



# nf
cd /home/chl/ppopp-code/nf_int/build/lonestar/analytics/gpu/sssp && make
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/delaunay_n20.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/msdoor.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/ldoor.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/audikw_1.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/flickr.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/rmat.3Mv.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/rmat.2Mv.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/USA-road-d.CAL.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/USA-road-d.USA.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/asia_osm.gr
CUDA_VISIBLE_DEVICES=7 ./sssp-gpu -s 0 /home/chl/ppopp-code/inputs/graph-int/circuit5M_dc.gr

# adds
cd /home/chl/ppopp-code/ads_int/ && make
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/delaunay_n20.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/msdoor.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/ldoor.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/audikw_1.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/flickr.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/rmat.3Mv.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/rmat.2Mv.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/USA-road-d.CAL.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/USA-road-d.USA.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/asia_osm.gr
CUDA_VISIBLE_DEVICES=7 ./sssp -s 0 /home/chl/ppopp-code/inputs/graph-int/circuit5M_dc.gr

# DL-BF
cd /home/chl/hbftest/HBF/HBF/build
device=6
times=1
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 150 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 140 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 40 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 400 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 3.5 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.1 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 PBCE_3000 3000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.USA.ddsg run V2 4 PBCE_3000 3000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/asia_osm.ddsg run V2 4 PBCE_3000 5000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/circuit5M_dc.ddsg run V2 8 BCE 2300 ${times} 0

# multiQueue
cd /home/chl/hbftest/HBF/HBF/build
device=7
times=1
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V3 16 BCE 1300 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V3 32 BCE 130 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V3 32 BCE 130 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V3 32 BCE 50 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V4 32 BCE 100 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V3 32 BCE 1.5 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V3 32 BCE 0.1 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V3 4 PBCE_3000 3000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.USA.ddsg run V3 4 PBCE_3000 3000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/asia_osm.ddsg run V3 4 PBCE_3000 5000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/circuit5M_dc.ddsg run V3 8 BCE 2300 ${times} 0

CUDA_VISIBLE_DEVICES=4 ./HBF /home/chl/data/USA-road-d.USA.ddsg run V3 4 BCE 1500 1 0 > res.txt


# delta
cd /home/chl/hbftest/HBF/HBF/build
device=7
times=1
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 delta 20000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 2500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 delta 2500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 delta 500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 delta 6000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 delta 60 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 delta 5 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 delta 350000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.USA.ddsg run V2 4 delta 200000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/asia_osm.ddsg run V2 4 delta 1500000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/circuit5M_dc.ddsg run V2 8 delta 23000 ${times} 0

CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 delta 30 5 0

# none
cd /home/chl/hbftest/HBF/HBF/build
device=7
times=1
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 none 10000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 none 140 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 none 130 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 none 50 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 none 750 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 none 3.6 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 none 0.6 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 none 3500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.USA.ddsg run V2 4 none 3500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/asia_osm.ddsg run V2 4 none 5000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/circuit5M_dc.ddsg run V2 8 none 2300 ${times} 0


# perfect
cd /home/chl/hbftest/HBF/HBF/build
device=6
times=1
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 perfect 1500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 perfect 140 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 perfect 130 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 perfect 50 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 perfect 650 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 perfect 3.6 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 perfect 0.6 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 perfect 3500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.USA.ddsg run V2 4 perfect 3500 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/asia_osm.ddsg run V2 4 perfect 5000 ${times} 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/circuit5M_dc.ddsg run V2 8 perfect 2300 ${times} 0

# BoostDijkstra
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/delaunay_n20.ddsg run BoostDijstra 16 BoostDijstra 1500 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/msdoor.ddsg run BoostDijstra 32 BoostDijstra 140 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/ldoor.ddsg run BoostDijstra 32 BoostDijstra 130 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/audikw_1.mtx run BoostDijstra 32 BoostDijstra 50 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/flickr.ddsg run BoostDijstra 32 BoostDijstra 650 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/rmat.3Mv.ddsg run BoostDijstra 32 BoostDijstra 3.6 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/rmat.2Mv.128Me run BoostDijstra 32 BoostDijstra 0.6 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run BoostDijstra 4 BoostDijstra 3500 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/USA-road-d.USA.ddsg run BoostDijstra 4 BoostDijstra 3500 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/asia_osm.ddsg run BoostDijstra 4 BoostDijstra 5000 1 0
CUDA_VISIBLE_DEVICES=6 ./HBF /home/chl/data/circuit5M_dc.ddsg run BoostDijstra 8 BoostDijstra 2300 1 0

bash bce.sh > bce.txt
bash pbce.sh > pbce.txt
bash delta.sh > delta.txt
bash none.sh > none.txt
bash perfect.sh > perfect.txt


device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 delta 20000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 2500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 delta 2500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 delta 500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 delta 3000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 delta 30 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 delta 5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 delta 350000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/USA-road-d.USA.ddsg run V2 4 delta 200000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/asia_osm.ddsg run V2 4 delta 1500000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/circuit5M_dc.ddsg run V2 8 delta 23000 8 0

# delaunay_n20
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1700 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1600 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1400 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1300 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1200 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1100 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 1000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 900 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 800 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/delaunay_n20.ddsg run V2 16 BCE 700 8 0

# msdoor
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 170 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 160 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 150 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 140 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 130 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 120 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 110 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 100 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 90 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 BCE 80 8 0

# del delta
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 8000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 3500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 2500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 1500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 1000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 800 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/msdoor.ddsg run V2 32 delta 300 8 0



# ldoor
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 170 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 160 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 150 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 140 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 130 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 120 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 110 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 100 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 90 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/ldoor.ddsg run V2 32 BCE 80 8 0

#audikw_1
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 100 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 90 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 80 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 70 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 60 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 50 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 40 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 30 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/audikw_1.mtx run V2 32 BCE 20 8 0

#flickr bash flickr.sh > flickr.txt &
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 2000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 1800 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 1600 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 1400 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 1200 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 1000 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 800 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 700 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 600 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 500 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 400 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 300 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 200 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/flickr.ddsg run V2 32 BCE 100 8 0

#rmat.3Mv.20Me
device=6
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 10 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 8 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 6 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 5.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 4 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 3.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 3 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 2.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 2 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 1.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 1 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 0.5 8 0

#rmat.2Mv.120Me
device=4
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 3 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 2.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 2 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 1.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 1 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.8 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.7 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.6 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.5 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.4 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.3 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.2 8 0
CUDA_VISIBLE_DEVICES=${device} ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.1 8 0

#USA-road-d.CAL
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 BCE 4000 8 0
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 BCE 3500 8 0
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 BCE 3000 8 0
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 BCE 2000 8 0
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 BCE 1500 8 0
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 BCE 1000 8 0

CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/rmat.3Mv.ddsg run V2 32 BCE 3.6 8 0
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/rmat.2Mv.128Me run V2 32 BCE 0.3 8 0

CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 PBCE_3000 3000 8 0
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 PBCE_3000 3000 8 0
CUDA_VISIBLE_DEVICES=3 ./HBF /home/chl/data/USA-road-d.CAL.ddsg run V2 4 delta 350000 8 0


CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/USA-road-d.USA.ddsg run V2 4 PBCE_3000 3000 8 0
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/asia_osm.ddsg run V2 4 PBCE_3000 5000 8 0
CUDA_VISIBLE_DEVICES=5 ./HBF /home/chl/data/circuit5M_dc.ddsg run V2 8 BCE 2300 8 0


CUDA_VISIBLE_DEVICES=7 ./HBF /home/chl/data/USA-road-d.CAL.gr cacValid V1 4 PBCE_3000 3000 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..USA-road-d.CAL.gr..PBCE
CUDA_VISIBLE_DEVICES=7 ./HBF /home/chl/data/USA-road-d.CAL.gr cacValid V1 4 normal 2000 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..USA-road-d.CAL.gr..BCE

CUDA_VISIBLE_DEVICES=7 ./HBF /home/chl/data/USA-road-d.USA.gr cacValid V1 4 PBCE_3000 3000 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..USA-road-d.USA.gr..PBCE &
CUDA_VISIBLE_DEVICES=7 ./HBF /home/chl/data/USA-road-d.USA.gr cacValid V1 4 normal 1500 100 1 0 > /home/chl/hbftest/HBF/HBF/nodeValid/nodeTime..cpu..USA-road-d.USA.gr..BCE &
