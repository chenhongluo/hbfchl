#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
namespace KernelV5
{
	template<int VW_SIZE, typename T>
	__device__ __forceinline__ void
		SWrite(thread_block_tile<VW_SIZE>& tile,
			T* writeQueueAddr, int * writeSizeAddr,
			int flag, T* data,
			int * queue, int &queueSize, int queueLimit,
			int &mymask
		)
	{
		unsigned mask = tile.ballot(flag);
		// devPrintfX(32, mask, "mask");

		int sum = __popc(mask);
		if (sum + queueSize > queueLimit)
		{
			// write to global mem if larger than shared mem
			int globalBias;
			if (tile.thread_rank() == 0)
				globalBias = atomicAdd(writeSizeAddr, queueSize);
			globalBias = tile.shfl(globalBias, 0);
			for (int j = tile.thread_rank(); j < founds; j += VW_SIZE)
				writeQueueAddr[globalBias + j] = queue[j];
			tile.sync();
			queueSize = 0;
		}
		if (flag)
		{
			// write to shared mem
			mask = mask & mymask;
			int pos = __popc(mask);
			queue[pos + queueSize] = data;
		}
		tile.sync();
		queueSize += sum;
	}

	template <int VW_SIZE>
	__global__ void HBFSearchV5Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF3, int* devF2,
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
		const int tileSharedLimit = (sharedLimit / 4) / blockdim * VW_SIZE;
		int* devF2Size = devSizes + 1;
		int relaxEdges = 0;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + g.thread_rank() / VW_SIZE * tileSharedLimit;
		int queueSize = 0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;

		//alloc node for warps tileID * tileLimit - tileID* tileLimit + tileLimit
		// 
		for (int i = tileID * tileLimit; i < devSizes[2]; i += IDStride * tileLimit)
		{
			for (int j = 0; j < tileLimit; j++) {
				if (i + j < devSizes[2]) {
					int index = devF3[i + j];
					int sourceWeight = devDistances[index].y;
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
						}
						SWrite< VW_SIZE, int>(tile, devF2, devF2Size, flag, dest.x, queue, queueSize, tileSharedLimit, mymask);
					}
				}
			}
		}
		SWrite< VW_SIZE, int>(tile, devF2, devF2Size, 0, dest.x, queue, queueSize, 0, mymask);
		if (tile.thread_rank() == 0) {
			atomicAdd(devSizes + 3, relaxEdges);
		}
	}

	template <int VW_SIZE>
	__global__ void selectNodesV5(
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2, int *devF3,
		int *__restrict__ devSizes,
		const int distanceLimit,
		const int sharedLimit,
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
		const int tileSharedLimit = (sharedLimit / 8) / blockdim * VW_SIZE;
		int* devF3Size = devSizes + 2;
		int* devF2Size = devSizes + 1;

		// for write in shared mem
		extern __shared__ int st[];
		int * st1 = st;
		int* st2 = st + (sharedLimit / 8);
		int *queue1 = st1 + g.thread_rank() / VW_SIZE * tileSharedLimit;
		int *queue2 = st2 + g.thread_rank() / VW_SIZE * tileSharedLimit; // bank conflict
		int queueSize1 = 0;
		int queueSize2 = 0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;
		// ��warp��
		for (int i = tileID; i < (devSizes[0] + VW_SIZE - 1) / VW_SIZE; i += IDStride)
		{
			int j = i * VW_SIZE + tile.thread_rank();
			int flag2 = 0;
			int flag3 = 0;
			if (j < devSizes[0]) {
				int index = devF1[j];
				if (devDistances[index].y <= distanceLimit) {
					flag3 = 1;
				}
				else {
					flag2 = 1;
					devDistances[index].x = level;
				}
			}
			SWrite< VW_SIZE, int>(tile, devF3, devF3Size, flag3, index, queue1, queueSize1, tileSharedLimit, mymask);
			SWrite< VW_SIZE, int>(tile, devF2, devF2Size, flag2, index, queue2, queueSize2, tileSharedLimit, mymask);
		}
		SWrite< VW_SIZE, int>(tile, devF3, devF3Size, 0, index, queue1, queueSize1, 0, mymask);
		SWrite< VW_SIZE, int>(tile, devF2, devF2Size, 0, index, queue2, queueSize2, 0, mymask);
	}
}

using namespace KernelV5;

#define kernelV5Atmoic64(vwSize,gridDim, blockDim, sharedLimit ,tileLimit) \
HBFSearchV5Atomic64<vwSize> << <gridDim, blockDim, sharedLimit >> > \
(devUpOutNodes, devUpOutEdges, devInt2Distances, devF3, devF2, devSizes, sharedLimit,tileLimit, level)

//user interface gridDim, blockDim, sharedLimit, devUpOutNodes, devUpOutEdges, devIntDistances, devInt2Distances, f1, f2, devSizes, sharedLimit,level
//name = {HBFSearchV5Atomic64,HBFSearchV5Atomic32}
//vwSize = 1,2,4,8,16,32
#define switchKernelV5(atomic64,vwSize,gridDim, blockDim, sharedLimit,tileLimit ) \
{\
	if (atomic64) {  \
		switch (vwSize) { \
		case 1:\
			kernelV5Atmoic64(1,gridDim, blockDim, sharedLimit,tileLimit);break; \
		case 2: \
			kernelV5Atmoic64(2,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 4: \
			kernelV5Atmoic64(4,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 8: \
			kernelV5Atmoic64(8,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 16: \
			kernelV5Atmoic64(16,gridDim, blockDim, sharedLimit,tileLimit); break;\
		case 32: \
			kernelV5Atmoic64(32,gridDim, blockDim, sharedLimit,tileLimit); break;\
		default: \
			__ERROR("no this vwsize")\
		}\
	} \
	else { \
		__ERROR("no atomic32") \
	}\
}

#define switchKernelV5Config(configs) \
	switchKernelV5(configs.atomic64,configs.vwSize,gridDim,blockDim,sharedLimit ,tileLimit)

#define selectNodesV5()
selectNodesV5<vwSize> << <gridDim, blockDim, sharedLimit >> > \
(devDistances, devF1, devF2, devF3, devSizes, distanceLimit, sharedLimit, level)
