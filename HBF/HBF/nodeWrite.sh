#！/bin/bash
logdir=/home/chl/hbftest/HBF/HBF/nodeWrite
cd /home/chl/hbftest/HBF/HBF/build
CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/USA-road-d.CAL.gr nodeWriteTest V0 1 none 0 1000000 > ${logdir}/V0.txt
CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/USA-road-d.CAL.gr nodeWriteTest V1 1 none 0 1000000 > ${logdir}/V1.txt
CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/USA-road-d.CAL.gr nodeWriteTest V2 1 none 0 1000000 > ${logdir}/V2.txt
CUDA_VISIBLE_DEVICES=1 ./HBF /home/chl/data/USA-road-d.CAL.gr nodeWriteTest V3 1 none 0 1000000 > ${logdir}/V3.txt