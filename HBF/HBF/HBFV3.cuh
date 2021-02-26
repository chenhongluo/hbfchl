#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
namespace KernelV3
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
	__global__ void HBFSearchV3Atomic64(
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
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int tileID = realID / WARPSIZE;
		const int IDStride = gridDim.x * (blockdim / WARPSIZE);
		const int warpSharedLimit = (sharedLimit / 4) / blockdim * WARPSIZE;
		int* devF2Size = devSizes + 1;
		int relaxEdges = 0;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + g.thread_rank() / WARPSIZE * warpSharedLimit;
		int queueSize = 0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;

		int kkk = 4;
		int ttt = devSizes[2] - devSizes[2] % kkk;
		for (int i = tileID * kkk; i < ttt; i += IDStride * kkk)
		{
			int	index0 = devF3[i];
			int	index1 = devF3[i + 1];
			int	index2 = devF3[i + 2];
			int	index3 = devF3[i + 3];
			int relaxSize0 = devNodes[index0 + 1] - devNodes[index0];
			int relaxSize1 = devNodes[index1 + 1] - devNodes[index1];
			int relaxSize2 = devNodes[index2 + 1] - devNodes[index2];
			int relaxSize3 = devNodes[index3 + 1] - devNodes[index3];
			int	sourceWeight0 = devDistances[index0].y;
			int	sourceWeight1 = devDistances[index1].y;
			int	sourceWeight2 = devDistances[index2].y;
			int	sourceWeight3 = devDistances[index3].y;
			int newWeight;
			int all = relaxSize0 + relaxSize1 + relaxSize2 + relaxSize3;
			// alloc edges in a warp
			for (int k = tile.thread_rank(); k < all + tile.thread_rank();
			 k += WARPSIZE)
			{
				//relax edge  if flag=1 write to devF2
				int flag = 0;
				int2 dest;
				int kk = 0;
				if (k<relaxSize0){
					kk = devNodes[index0] + k;
					dest = devEdges[kk];
					newWeight = sourceWeight0 + dest.y;
				}
				else if( k < relaxSize0 + relaxSize1){
					kk = devNodes[index1] + k - relaxSize0;
					dest = devEdges[kk];
					newWeight = sourceWeight1 + dest.y;
				}else if(k < relaxSize0 + relaxSize1 + relaxSize3){
					kk = devNodes[index2] + k - relaxSize0 - relaxSize1;
					dest = devEdges[kk];
					newWeight = sourceWeight2 + dest.y;
				}else if(k < all){
					kk = devNodes[index3] + k - relaxSize0 - relaxSize1 -relaxSize2;
					dest = devEdges[kk];
					newWeight = sourceWeight3 + dest.y;
				}
				tile.sync();
				if(k < all){
					int2 toWrite = make_int2(level, newWeight);
					unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
						reinterpret_cast<unsigned long long &>(toWrite));
					int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
					flag = ((oldNode2Weight.y > newWeight) && (level > oldNode2Weight.x));
				}
				SWrite<WARPSIZE, int>(tile, devF2, devF2Size, flag, dest.x, queue, queueSize, warpSharedLimit, mymask);
			}
		}
		SWrite< WARPSIZE, int>(tile, devF2, devF2Size, 0, 0, queue, queueSize, 0, mymask);
		// if (tile.thread_rank() == 0) {
		// 	atomicAdd(devSizes + 3, relaxEdges);
		// }
	}

	template <int VW_SIZE>
	__global__ void SelectNodesV3(
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

using namespace KernelV3;

#define selectNodesV3(configs) \
SelectNodesV3<WARPSIZE> <<<gdim, bdim, sharedLimit>>> \
(devInt2Distances,devF1, devF2, devF3, devSizes, distanceLimit, sharedLimit,level);

#define kernelV3Atmoic64(vwSize,gridDim, blockDim,sharedLimit) \
HBFSearchV3Atomic64<vwSize> << <gridDim, blockDim, sharedLimit >> > \
(devUpOutNodes, devUpOutEdges, devInt2Distances, devF3, devF2, devSizes, sharedLimit, level)

#define switchKernelV3(atomic64,vwSize,gridDim, blockDim,  sharedLimit ) \
{\
	if (atomic64) {  \
		switch (vwSize) { \
		case 1:\
			kernelV3Atmoic64(1,gridDim, blockDim, sharedLimit);break; \
		case 2: \
			kernelV3Atmoic64(2,gridDim, blockDim, sharedLimit); break;\
		case 4: \
			kernelV3Atmoic64(4,gridDim, blockDim, sharedLimit); break;\
		case 8: \
			kernelV3Atmoic64(8,gridDim, blockDim, sharedLimit); break;\
		case 16: \
			kernelV3Atmoic64(16,gridDim, blockDim, sharedLimit); break;\
		case 32: \
			kernelV3Atmoic64(32,gridDim, blockDim, sharedLimit); break;\
		default: \
			__ERROR("no this vwsize")\
		}\
	} \
	else { \
		__ERROR("no atomic32") \
	}\
}

#define switchKernelV3Config(configs) \
	switchKernelV3(configs.atomic64,vwSize,gdim, bdim,sharedLimit)