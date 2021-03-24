#pragma once

#include <cub/cub.cuh>
struct gpu_mutex{
	int* lockv;
	gpu_mutex(){
		cudaMalloc((void**)&lockv, 1 * sizeof(int));
		int v = 0;
		cudaMemcpy(lockv, &v, 1 * sizeof(int), cudaMemcpyHostToDevice);
	}
	__device__ void lock(){
		while(atomicCAS(lockv,0,1) != 0);
	}

	__device__ void unlock(){
		atomicExch(lockv,0);
	}
};

