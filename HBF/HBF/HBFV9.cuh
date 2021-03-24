#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "gbar.cuh"
using namespace cooperative_groups;

#define WARPSIZE 32
namespace KernelV9
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

	template<typename T>
	__device__ __forceinline__ void
		AWrite(T* writeQueueAddr, int * writeSizeAddr,
			int flag, T data)
	{
		if(flag){
			int pos = atomicAdd(writeSizeAddr,1);
			writeQueueAddr[pos] = data;
		}
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
		// devPrintfX(WARPSIZE, mask, "mask");

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

	__device__ void swapPtr(int* & a,int* & b){
		int * c = b;
		b = a;
		a = c;
	}

	__device__ void SelectNodesV9(
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2, int* devF3,
		int *__restrict__ devSizes,
		int F1index,int F2index,int F3index,
		const float distanceLimit,
		const int tileSharedLimit,
		int realID,
		thread_block_tile<WARPSIZE> tile,
		int* queue1,
		int* queue2,
		unsigned mymask,
		int level)
	{
		int queueSize1 = 0;
		int queueSize2 = 0;
		int* devF1Size = devSizes + 0 + F1index;
		int* devF2Size = devSizes + 2 + F2index;
		int* devF3Size =  devSizes + 4 + F3index;
		const int warpID = realID / WARPSIZE;
		const int warpStride = gridDim.x * (blockDim.x / WARPSIZE);
		
		for (int i = warpID; i < (*devF1Size + WARPSIZE - 1) / WARPSIZE; i += warpStride)
		{
			int j = i * WARPSIZE + tile.thread_rank();
			int flag1 = 0;
			int flag2 = 0;
			int index = -1;
			if (j < *devF1Size) {
				index = devF1[j];
				if (devDistances[index].y <= distanceLimit) {
					flag1 = 1;
				}
				else {
					flag2 = 1;
					devDistances[index].x = level;
				}
			}
			
			SWrite<WARPSIZE, int>(tile, devF3, devF3Size, flag1, index, queue1, queueSize1, tileSharedLimit, mymask);
			SWrite<WARPSIZE, int>(tile, devF2, devF2Size, flag2, index, queue2, queueSize2, tileSharedLimit, mymask);
		}
		SWrite<WARPSIZE, int>(tile, devF3, devF3Size, 0, 0, queue1, queueSize1, 0, mymask);
		SWrite<WARPSIZE, int>(tile, devF2, devF2Size, 0, 0, queue2, queueSize2, 0, mymask);
	}

	__global__ void __launch_bounds__(256) HBFSearchV9Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2, int*devF3,
		int *__restrict__ devSizes,
		const int sharedLimit,
		float initDL,
		float dl,
		GlobalBarrier gb)
	{
		thread_block g = this_thread_block();
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();

		float distanceLimit = initDL;
		// for write in shared mem
		extern __shared__ int sharedPtr[];
		const int warpSharedLimit1 = (sharedLimit / 4) / blockdim * WARPSIZE;
		const int warpSharedLimit2 = (sharedLimit / 8) / blockdim * WARPSIZE;
		int *queue = sharedPtr + g.thread_rank() / WARPSIZE * warpSharedLimit1;
		int *queue1 = sharedPtr + g.thread_rank() / WARPSIZE * warpSharedLimit2;
		int *queue2 = sharedPtr + (sharedLimit / 8) + g.thread_rank() / WARPSIZE * warpSharedLimit2;
		unsigned mymask = (1 << tile.thread_rank()) - 1;

		int F1index = 0;
		int F2index = 0;
		int F3index = 0;
		int level = 0;
		int relaxNodes = 0;


		while(devSizes[F1index] > 0){
			level = level +1;
			
			// gb.Sync();
			// if(realID == 0){
			// 	printf("00000:%d,%d,%d,%d\n",devSizes[0],devSizes[2],devSizes[4],level);
			// }
			// SelectNodesV9(devDistances,devF1, devF2, devF3, devSizes,F1index,F2index,F3index,
			// 	distanceLimit,warpSharedLimit2,realID,tile,queue1,queue2,mymask,level);
			// gb.Sync();
			// if(realID == 0){
			// 	printf("11111:%d,%d,%d,%d\n",devSizes[0],devSizes[2],devSizes[4],level);
			// }

			// distanceLimit += dl;
			if(devSizes[5] < 8000){
				distanceLimit += dl;
			}
			relaxNodes = 0;
			gb.Sync();
			if(realID == 0){
				devSizes[5] = 0;
			}

			// if(devSizes[4+F3index] == 0){
			// 	distanceLimit += dl;
			// }
			int vwSize = 4;
			const int tileID = realID / vwSize;
			const int tileThreadId = realID % vwSize;
			const int IDStride = gridDim.x * (blockdim / vwSize);
			int* devF1Size = devSizes + 0 + F1index;
			int* devF2Size = devSizes + 2 + F2index;
			for (int i = tileID; i < *devF1Size; i += IDStride)
			{
				int index = devF1[i];
				int sourceWeight = devDistances[index].y;
				if(sourceWeight > distanceLimit){
					if(tileThreadId == 0){
						int2 toWrite = make_int2(level, sourceWeight);
						unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[index]),
								reinterpret_cast<unsigned long long &>(toWrite));
						int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
						int flag = (level > oldNode2Weight.x);
						AWrite<int>(devF2,devF2Size,flag,index);
					}
					continue;
				}
				if(tileThreadId == 0){
					relaxNodes += 1;
				}
				int nodeS = devNodes[index];
				int nodeE = devNodes[index + 1];
				// alloc edges in a warp
				for (int k = nodeS + tileThreadId; k < nodeE + tileThreadId; k += vwSize)
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
						AWrite<int>(devF2,devF2Size,flag,dest.x);
					}
				}
			}
			gb.Sync();
			// if(realID == 0){
			// 	printf("22222:%d,%d,%d,%d\n",devSizes[0],devSizes[2],devSizes[4],level);
			// }
			swapPtr(devF1,devF2);

			if(realID == 0){
				// relaxNodes += devSizes[4 + F3index];
				devSizes[F1index] = devSizes[2+F2index];
				devSizes[2+F2index] = 0;
				devSizes[4+F3index] = 0;
			}

			if(tileThreadId == 0){
				atomicAdd(devSizes + 5,relaxNodes);
				atomicAdd(devSizes+6,relaxNodes);
			}
			
			// if(realID == 0){
			// 	printf("33333:%d,%d,%d,%d\n",devSizes[0],devSizes[2],devSizes[4],level);
			// }
			gb.Sync();
		}

		if(realID == 0){
			atomicAdd(devSizes+7,level);
		}
	}	
}

using namespace KernelV9;

#define kernelV9Atmoic64(gdim, bdim,sharedLimit,initDL,distanceLimit,gb) \
HBFSearchV9Atomic64<<<gdim, bdim, sharedLimit >>> \
(devUpOutNodes, devUpOutEdges, devInt2Distances, devF1, devF2,devF3, devSizes, sharedLimit, initDL,distanceLimit,gb)