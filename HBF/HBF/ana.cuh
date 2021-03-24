#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
namespace KernelValid
{
	template <int VW_SIZE>
	__global__ void cacValidKernel(
		int *__restrict__ devNodes,
		int2 *__restrict__ devDistances,
		int2 *__restrict__ devTrueDistances,
		int* devF3, 
		int *__restrict__ devSizes,
		int2 *validRes,
		int* validSizes,
		int level)
	{
		//alloc node&edge
		thread_block g = this_thread_block();
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int tileID = realID / VW_SIZE;
		const int tileThreadId = realID % VW_SIZE;
		const int IDStride = gridDim.x * (blockdim / VW_SIZE);
		int v = 0,vd = 0;
		int nv = 0,nvd =0;
		for (int i = tileID; i < devSizes[2]; i += IDStride)
		{
			int index = devF3[i];
			int d = devDistances[index].y;
			int flag = 0;
			if(d == devTrueDistances[index].y){
				flag = 1;
				v ++;
				vd += (devNodes[index + 1] - devNodes[index]);
			}else{
				flag = 0;
				nv ++;
				nvd += (devNodes[index + 1] - devNodes[index]);
			}
			tile.sync();
			int globalBias = atomicAdd(validSizes, 1);
			validRes[globalBias] = make_int2(d,flag);
		}
		atomicAdd(validSizes+1, v);
		atomicAdd(validSizes+2, nv);
		atomicAdd(validSizes+3, vd);
		atomicAdd(validSizes+4, nvd);
	}
}

using namespace KernelValid;

#define switchKernelCacValid() \
cacValidKernel<1><<<gdim, bdim>>> \
(devUpOutNodes,devInt2Distances, devTrueInt2Distances, devF3, devSizes,validRes,validSizes, level);