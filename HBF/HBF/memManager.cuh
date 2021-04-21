#pragma once

#include <algorithm>
#include <cub/cub.cuh>
#include  <vector>
using namespace std;
#include "mutex.cuh"
struct memManager{
	int* mem;
	int* size;
	int* queue;
	// gpu_mutex spinlock;
	int blockSize;
	int bsize;
	memManager(int qsize,int blockSize){
		qsize = qsize - qsize % blockSize;
		this->blockSize = blockSize;
		int bsize = qsize/blockSize;
		this->bsize = bsize;
		cudaMalloc((void**)&size, 1 * sizeof(int));
		cudaMalloc((void**)&mem, qsize * sizeof(int));
		cudaMalloc((void**)&queue, bsize * sizeof(int));
		vector<int> vs(bsize,0);
		for(int i=0;i<vs.size();i++){
			vs[i] = i;
		}
		cudaMemcpy(size, &bsize, 1 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(queue, &vs[0], bsize * sizeof(int), cudaMemcpyHostToDevice);
	}

	void debug(){
		int vvvv = 0;
		cudaMemcpy(&vvvv, size, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		vector<int> vs(bsize,0);
		cudaMemcpy(&vs[0], queue, bsize * sizeof(int), cudaMemcpyDeviceToHost);
		// std::sort(vs.begin(),vs.end());
		vector<int> vvs(blockSize,0);
		cudaMemcpy(&vvs[0], getDevicePoint(2047), blockSize * sizeof(int), cudaMemcpyDeviceToHost);
		cout<<vvvv<<" "<<vs[bsize-1] <<" "<< vvs[0]<<vvs[1]<<vvs[2]<<endl;
	}

	__host__ __device__ int getData(int posi,int posj){
		return mem[posi*blockSize+posj];
	}

	__host__ __device__ void setData(int posi,int posj,int data){
		mem[posi*blockSize+posj] = data;
	}

	__host__ __device__ int* getDevicePoint(int v){
		return mem + v * blockSize;
	}
	__device__ int getBlock(){
		// spinlock.lock();
		int pos = atomicSub(size,1);
		int ret = *(int volatile*)(queue+pos);
		// spinlock.unlock();
		return ret;
	}

	__device__ void release (int qs){
		int pos = atomicAdd(size,1);
		*(int volatile*)(queue+pos) = qs;
	}
};

