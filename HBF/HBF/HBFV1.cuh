#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
namespace Kernels
{
	template <int VW_SIZE>
	__global__ void HBFSearchV1Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2,
		int *__restrict__ devSizes,
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
		const int tileSharedLimit =  sharedLimit / VW_SIZE;
		int* devF2Size = devSizes + 1;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + g.thread_rank() / VW_SIZE * tileSharedLimit;
		int founds = 0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;
		int globalBias;

		//alloc node for warps
		for (int i = tileID; i < devSizes[0]; i += IDStride)
		{
			int index = devF1[i];
			int sourceWeight = devDistances[index].y;
			// devPrintf(1, sourceWeight, "sourceWeight");
			// devPrintf(128, tile.thread_rank(), "tile.thread_rank()");
			int nodeS = devNodes[index];
			int nodeE = devNodes[index + 1];
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
				unsigned mask = tile.ballot(flag);
				// devPrintfX(32, mask, "mask");

				int sum = __popc(mask);
				if (sum + founds > tileSharedLimit)
				{
					// write to global mem if larger than shared mem
					if (tile.thread_rank() == 0)
						globalBias = atomicAdd(devF2Size, founds);
					globalBias = tile.shfl(globalBias, 0);
					for (int j = tile.thread_rank(); j < founds; j += VW_SIZE)
						devF2[globalBias + j] = queue[j];
					tile.sync();
					founds = 0;
				}
				if (flag)
				{
					// write to shared mem
					mask = mask & mymask;
					int pos = __popc(mask);
					queue[pos + founds] = dest.x;
				}
				tile.sync();
				founds += sum;
			}
		}
		// write to global mem
		if (tile.thread_rank() == 0)
			globalBias = atomicAdd(devSizes + 1, founds);
		globalBias = tile.shfl(globalBias, 0);
		for (int j = tile.thread_rank(); j < founds; j += 32)
			devF2[globalBias + j] = queue[j];
	}
}