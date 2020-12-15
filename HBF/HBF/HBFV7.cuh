#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
#define bulkSize 10

namespace KernelV7
{
	template <int VW_SIZE>
	__global__ void minmax(
		int2 *__restrict__ devDistances,
		int* devF1,
		int *__restrict__ devSizes,
		int *mm,
		int level)
	{
		int mi = INT_MAX, ma = 0;
		const int realID = threadIdx.x + blockIdx.x * gridDim.x;
		const int count = blockDim.x * gridDim.x;
		for (int i = realID; i < devSizes[0]; i += count) {
			if (devDistances[devF1[i]].y < mi) {
				mi = devDistances[devF1[i]].y;
			}
			//__syncwarp(0xFFFFFFFF);
			if (devDistances[devF1[i]].y > ma) {
				ma = devDistances[devF1[i]].y;
			}
		}
		if (mi != INT_MAX) {
			atomicMin(mm, mi);
		}
		if (ma != 0) {
			atomicMax(mm + 1, ma);
		}
	}
	template <int VW_SIZE>
	__global__ void toBulk(
		int2 *__restrict__ devDistances,
		int* devF1,
		int *__restrict__ devSizes,
		int *mm,
		int level) 
	{
		int temp[bulkSize];
#pragma unroll
		for (int j = 0; j < bulkSize; j++) {
			temp[j] = 0;
		}
		const int realID = threadIdx.x + blockIdx.x * gridDim.x;
		const int count = blockDim.x * gridDim.x;
		int d = (mm[1] - mm[0]) / bulkSize + 1;
		for (int i = realID; i < devSizes[0]; i += count) {
			int k = (devDistances[devF1[i]].y - mm[0]) / d;
#pragma unroll
			for (int j = 0; j < bulkSize; j++) {
				if (j == k) {
					temp[j]++;
				}
			}
		}
#pragma unroll
		for (int j = 0; j < bulkSize; j++) {
			atomicAdd(mm + 2 + j, temp[j]);
		}
	}
}

using namespace KernelV7;

#define getMinMax() \
minmax<WARPSIZE><<<gdim, bdim>>> \
(devInt2Distances, devF1, devSizes, devMM, level);

#define getBulk() \
toBulk<WARPSIZE><<<gdim, bdim>>> \
(devInt2Distances, devF1, devSizes, devMM, level);
