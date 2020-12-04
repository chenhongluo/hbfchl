#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

namespace KernelV2
{
	template <int VW_SIZE, typename T>
	__device__ __forceinline__ void
		VWInclusiveScanAdd(thread_block_tile<VW_SIZE> &tile, const T &value, T &sum)
	{
		sum = value;
		for (int i = 1; i <= tile.size() / 2; i *= 2)
		{
			T n = tile.shfl_up(sum, i);
			if (tile.thread_rank() >= i)
			{
				sum += n;
			}
		}
	}

	template<int VW_SIZE, typename T>
	__device__ __forceinline__ void
		VWWrite(thread_block_tile<VW_SIZE>& tile, int *pAllsize, T* writeStartAddr, const int& writeCount, T* data)
	{
		int sum = 0;
		int bias = 0;
		VWInclusiveScanAdd<VW_SIZE, int>(tile, writeCount, sum);
		// devPrintfInt(32,sum,"sum");
		if (tile.thread_rank() == tile.size() - 1)
		{
			bias = atomicAdd(pAllsize, sum);
		}
		bias = tile.shfl(bias, tile.size() - 1);
		sum -= writeCount;

		for (int it = 0; it < writeCount; it++)
		{
			writeStartAddr[bias + sum + it] = data[it];
		}
	}

	template <int VW_SIZE>
	__global__ void HBFSearchV2Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2,
		int *__restrict__ devSizes,
		const int sharedLimit,
		const int tileLimit,
		int level)
	{
		//alloc node&edge
		thread_block g = this_thread_block();
		thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int tileID = realID / VW_SIZE;
		const int IDStride = gridDim.x * (blockdim / VW_SIZE);
		const int threadLimit = (sharedLimit / 4) / blockdim;
		int* devF2Size = devSizes + 1;
		int relaxEdges = 0;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + threadLimit * g.thread_rank();
		int founds = 0;

		//alloc node for warps
		for (int i = tileID * tileLimit; i < devSizes[0]; i += IDStride * tileLimit)
		{
			for (int j = 0; j < tileLimit; j++) {
				if (i + j < devSizes[0]) {
					int index = devF1[i + j];
					int sourceWeight = devDistances[index].y;
					// devPrintf(1, sourceWeight, "sourceWeight");
					// devPrintf(128, tile.thread_rank(), "tile.thread_rank()");
					int nodeS = devNodes[index];
					int nodeE = devNodes[index + 1];
					relaxEdges += (nodeE - nodeS);
					// alloc edges in a warp
					for (int k = nodeS + tile.thread_rank(); k < nodeE + tile.thread_rank(); k += VW_SIZE)
					{
						//relax edge  if flag=1 write to devF2
						int flag = 0;
						int2 dest;
						if (k < nodeE)
						{
							dest = devEdges[k];
							int newWeight = sourceWeight + dest.y;
							int2 toWrite = make_int2(level, newWeight);
							unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
								reinterpret_cast<unsigned long long &>(toWrite));
							int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
							flag = ((oldNode2Weight.y > newWeight) && (level > oldNode2Weight.x));
							if (flag) {
								queue[founds++] = dest.x;
							}
						}
						if (tile.any(founds >= threadLimit)) {
							VWWrite<VW_SIZE, int>(tile, devF2Size, devF2, founds, queue);
							founds = 0;
						}
					}
				}
			}
		}
		// write to global mem
		VWWrite<VW_SIZE, int>(tile, devF2Size, devF2, founds, queue);
		if (tile.thread_rank() == 0) {
			atomicAdd(devSizes + 2, relaxEdges);
		}
	}
}

using namespace KernelV2;

#define kernelV2Atmoic64(vwSize,gridDim, blockDim, sharedLimit ,tileLimit) \
HBFSearchV2Atomic64<vwSize> << <gridDim, blockDim, sharedLimit >> > \
(devUpOutNodes, devUpOutEdges, devInt2Distances, devF1, devF2, devSizes, sharedLimit,tileLimit, level)

//user interface gridDim, blockDim, sharedLimit, devUpOutNodes, devUpOutEdges, devIntDistances, devInt2Distances, f1, f2, devSizes, sharedLimit,level
//name = {HBFSearchV2Atomic64,HBFSearchV2Atomic32}
//vwSize = 1,2,4,8,16,32
#define switchKernelV2(atomic64,vwSize,gridDim, blockDim, sharedLimit,tileLimit ) \
{\
	if (atomic64) {  \
		switch (vwSize) { \
		case 1:\
			kernelV2Atmoic64(1,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 2: \
			kernelV2Atmoic64(2,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 4: \
			kernelV2Atmoic64(4,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 8: \
			kernelV2Atmoic64(8,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 16: \
			kernelV2Atmoic64(16,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 32: \
			kernelV2Atmoic64(32,gridDim, blockDim, sharedLimit,tileLimit); break;\
		default: \
			__ERROR("no this vwsize")\
		}\
	} \
	else { \
		__ERROR("no atomic32") \
	}\
}

#define switchKernelV2Config(configs) \
	switchKernelV2(configs.atomic64,configs.vwSize,gdim, bdim, sharedLimit ,tileLimit)
