#pragma once

#include <algorithm>
#include <cub/cub.cuh>
#include "memManager.cuh"
#include "mutex.cuh"
#include  <vector>
using namespace std;
// struct WorkLists{
// 	int size;
// 	int blockSize;
// 	int* metas;
// 	int* metaSizes;
// 	int* qsizes;

// 	memManager mem;
// 	mutexs ms;
// 	WorkLists(int _size,int _blockSize,int v):ms(_size),mem(v*5,1024*160){
// 		this->size = _size;
// 		this->blockSize = _blockSize;
// 		int metaSize = (v + blockSize -1)/blockSize;
// 		cudaMalloc((void**)&metas, metaSize * sizeof(int));
// 		cudaMalloc((void**)&qsizes, size * sizeof(int));
// 		cudaMalloc((void**)&metaSizes, size * sizeof(int));
// 		vector<int> vs(size,0);
// 		cudaMemcpy(qsizes, &vs[0], size * sizeof(int), cudaMemcpyHostToDevice);
// 		cudaMemcpy(metaSizes, &vs[0], size * sizeof(int), cudaMemcpyHostToDevice);
// 	}

// 	~WorkLists(){
// 		cudaFree(metas);
// 		cudaFree(metaSizes);
// 		cudaFree(qsizes);
// 	}

// 	__device__ int getData(int i,int j){
// 		int posii = j / blockSize;
// 		int posi = metas[posii];
// 		int posj = j % blockSize;
// 		return mem.getData(posi,posj);
// 	}

// 	__device__ int getPos(int i,int k){
// 		return atomicAdd(qsizes+i,k);
// 	}

// 	__device__ void setData(int i,int j,int data){
// 		int maxSize = metaSizes[i] * blockSize;
// 		while(j >= maxSize){
// 			ms.lock(i);
// 			maxSize = *(int volatile*)(metaSizes+i) * blockSize;
// 			if(j >= maxSize){
// 				metas[metaSize[i]] t = mem.getBlock();
// 				atomicAdd(metaSizes+i,1);
// 			}
// 			ms.unlock(i);
// 		}
// 		int posii = j / blockSize;
// 		// int posi = metas[posii];
// 		int posi = *(int volatile*)(metaSizes+posii);
// 		int posj = j % blockSize;
// 		mem.setData(posi,posj,data);
// 	}

// 	__device__ void clear(int i){
// 		int realID = blockIdx.x * gridDim.x + threadIdx.x;
// 		int nthreads = gridDim.x * blockDim.x;
// 		int csize = *(int volatile*)(metaSize + i);
// 		for(int i = realID,i<csize;i+=nthreads){
// 			int t = metas[i];
// 			mem.release(t);
// 		}
// 	}
// };

struct WorkListsPlain{
	int size;
	int* data;
	int* qsizes;
	int v;
	WorkListsPlain(int _size,int _v){
		this->size = _size;
		this->v = _v;
		cudaMalloc((void**)&data, size * v * sizeof(int));
		cudaMalloc((void**)&qsizes, size * 32 * sizeof(int));
		vector<int> vs(size * 32,0);
		cudaMemcpy(qsizes, &vs[0], size * 32 * sizeof(int), cudaMemcpyHostToDevice);
	}

	__device__ int getData(int i,int j){
		return data[i*v+j];
	}

	__device__ int getPos(int i,int k){
		return atomicAdd(qsizes+i*32,k);
	}

	__device__ void setData(int i,int j,int d){
		data[i*v+j] = d;
	}

	__device__ int getMaxSizeOfQueue(int i){
		return qsizes[i*32];
	}

	__device__ void clear(int i){
		qsizes[i*32] = 0;
	}

	void deleteData(){
		cudaFree(data);
		cudaFree(qsizes);
	}
};