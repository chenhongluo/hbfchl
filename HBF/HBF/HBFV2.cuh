#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
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

	template <int VW_SIZE, typename T>
	__device__ __forceinline__ void
	VWInclusiveScanMax(thread_block_tile<VW_SIZE> &tile, const T &value, T &maxValue)
	{
		maxValue = value;
		for (int i = 1; i <= tile.size() / 2; i *= 2)
		{
			T n = tile.shfl_up(maxValue, i);
			if (tile.thread_rank() >= i)
			{
				maxValue = maxValue<n?n:maxValue;
			}
		}
		maxValue = tile.shfl(maxValue, tile.size() - 1);
	}

	template<int VW_SIZE, typename T>
	__device__ __forceinline__ void
		SWrite(thread_block_tile<VW_SIZE>& tile,
			T* writeQueueAddr, int * writeSizeAddr,
			int flag, T data,
			int * queue, int &queueSize, const int queueLimit,
			unsigned &mymask)
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
			for (int j = tile.thread_rank(); j < queueSize; j += VW_SIZE)
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
	__global__ void HBFSearchV2Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF3, int* devF2,
		int *__restrict__ devSizes,
		const int sharedLimit,
		int level)
	{
		//alloc node&edge
		thread_block g = this_thread_block();
		thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
		thread_block_tile<WARPSIZE> tile2 = tiled_partition<WARPSIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int tileID = realID / VW_SIZE;
		const int IDStride = gridDim.x * (blockdim / VW_SIZE);
		const int warpSharedLimit = (sharedLimit / 4) / blockdim * WARPSIZE;
		int* devF2Size = devSizes + 1;
		int relaxEdges = 0;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + g.thread_rank() / WARPSIZE * warpSharedLimit;
		int queueSize = 0;
		unsigned mymask = (1 << tile2.thread_rank()) - 1;

		int vwn = WARPSIZE/VW_SIZE;
		int kkk = devSizes[2];
		if(kkk % vwn != 0)
			kkk = kkk + (vwn - kkk % vwn);
		for (int i = tileID; i < kkk; i += IDStride)
		{
			int index,sourceWeight,nodeS = 0,nodeE = 0,relaxSize = 0,maxRelaxSize = 0;
			if(i < devSizes[2]){
				index = devF3[i];
				sourceWeight = devDistances[index].y;
				nodeS = devNodes[index];
				nodeE = devNodes[index + 1];
				relaxSize = nodeE - nodeS;
				maxRelaxSize = 0;
				relaxEdges += relaxSize;
			}
			VWInclusiveScanMax<WARPSIZE,int>(tile2,relaxSize,maxRelaxSize);
			// alloc edges in a warp
			for (int k = nodeS + tile.thread_rank(); k < nodeS + maxRelaxSize + tile.thread_rank();
			 k += VW_SIZE)
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
				SWrite<WARPSIZE, int>(tile2, devF2, devF2Size, flag, dest.x, queue, queueSize, warpSharedLimit, mymask);
			}
		}
		SWrite< WARPSIZE, int>(tile2, devF2, devF2Size, 0, 0, queue, queueSize, 0, mymask);
		// if (tile.thread_rank() == 0) {
		// 	atomicAdd(devSizes + 3, relaxEdges);
		// }
	}

	template <int VW_SIZE>
	__global__ void SelectNodesV2(
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2, int* devF3,
		int *__restrict__ devSizes,
		const float distanceLimit,
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
		for (int i = tileID; i < (devSizes[0] + VW_SIZE - 1) / VW_SIZE; i += IDStride)
		{
			int j = i * VW_SIZE + tile.thread_rank();
			int flag2 = 0;
			int flag3 = 0;
			int index = -1;
			if (j < devSizes[0]) {
				index = devF1[j];
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
		SWrite< VW_SIZE, int>(tile, devF3, devF3Size, 0, 0, queue1, queueSize1, 0, mymask);
		SWrite< VW_SIZE, int>(tile, devF2, devF2Size, 0, 0, queue2, queueSize2, 0, mymask);
	}
}

using namespace KernelV2;

#define selectNodesV2(configs) \
SelectNodesV2<WARPSIZE> <<<gdim, bdim, sharedLimit>>> \
(devInt2Distances,devF1, devF2, devF3, devSizes, distanceLimit, sharedLimit,level);

#define kernelV2Atmoic64(vwSize,gridDim, blockDim,sharedLimit) \
HBFSearchV2Atomic64<vwSize> << <gridDim, blockDim, sharedLimit >> > \
(devUpOutNodes, devUpOutEdges, devInt2Distances, devF3, devF2, devSizes, sharedLimit, level)

#define switchKernelV2(atomic64,vwSize,gridDim, blockDim,  sharedLimit ) \
{\
	if (atomic64) {  \
		switch (vwSize) { \
		case 1:\
			kernelV2Atmoic64(1,gridDim, blockDim, sharedLimit);break; \
		case 2: \
			kernelV2Atmoic64(2,gridDim, blockDim, sharedLimit); break;\
		case 4: \
			kernelV2Atmoic64(4,gridDim, blockDim, sharedLimit); break;\
		case 8: \
			kernelV2Atmoic64(8,gridDim, blockDim, sharedLimit); break;\
		case 16: \
			kernelV2Atmoic64(16,gridDim, blockDim, sharedLimit); break;\
		case 32: \
			kernelV2Atmoic64(32,gridDim, blockDim, sharedLimit); break;\
		default: \
			__ERROR("no this vwsize")\
		}\
	} \
	else { \
		__ERROR("no atomic32") \
	}\
}

#define switchKernelV2Config(configs) \
	switchKernelV2(configs.atomic64,vwSize,gdim, bdim,sharedLimit)