#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
// Primary header is compatible with pre-C++11, collective algorithm headers require C++11
#include <cooperative_groups.h>
// Optionally include for memcpy_async() collective
#include <cooperative_groups/memcpy_async.h>
// Optionally include for reduce() collective
#include <cooperative_groups/reduce.h>
using namespace cooperative_groups;

#define BLOCKDIM 32
#define SHAREDLIMIT 32
namespace Kernels
{
	template <int VW_SIZE>
	__global__ void HBFSearchV0Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF1,int* devF2,
		int *__restrict__ devSizes,
		int level)
	{
		//alloc node&edge
		thread_block g = this_thread_block();
		thread_block_tile<VW_SIZE> tile = tiled_partition<VW_SIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int RealID = g.group_index().x * BLOCKDIM + g.thread_rank();
		const int tileID = RealID / VW_SIZE;
		const int IDStride = gridDim.x * (BLOCKDIM / VW_SIZE);

		// for write in shared mem
		__shared__ int st[SHAREDLIMIT * BLOCKDIM / 32];
		int *queue = st + g.thread_rank() / 32 * SHAREDLIMIT;
		int Founds = 0;
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
			// __syncwarp(0xFFFFFFFF);
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
				if (sum + Founds > SHAREDLIMIT)
				{
					// write to global mem if larger than shared mem
					if (tile.thread_rank() == 0)
						globalBias = atomicAdd(devSizes+1, Founds);
					globalBias = tile.shfl(globalBias, 0);
					for (int j = tile.thread_rank(); j < Founds; j += 32)
						devF2[globalBias + j] = queue[j];
					tile.sync();
					Founds = 0;
				}
				if (flag)
				{
					// write to shared mem
					mask = mask & mymask;
					int pos = __popc(mask);
					queue[pos + Founds] = dest.x;
				}
				tile.sync();
				Founds += sum;
			}
		}
		// write to global mem
		if (tile.thread_rank() == 0)
			globalBias = atomicAdd(devSizes + 1, Founds);
		globalBias = tile.shfl(globalBias, 0);
		for (int j = tile.thread_rank(); j < Founds; j += 32)
			devF2[globalBias + j] = queue[j];
	}
}