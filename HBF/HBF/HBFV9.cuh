#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "gbar.cuh"
#include "internal.h"
using namespace cooperative_groups;

#define WARPSIZE 32
#define __tb_sssp_kernel 256
#define DEGREE_LIMIT 32
#define TB_SIZE 256

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

	__device__ int roundup(int data, int mult) {
		auto rem = data % mult;
	  
		if (!rem)
		  return data;
		return data + (mult - rem);
	  }

	__device__ void sssp_kernel_dev(
		unsigned tid,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2, int*devF3,
		int *__restrict__ devSizes,int level,int vwSize
	)
	{
		const unsigned __kernel_tb_size = __tb_sssp_kernel;
		typedef int index_type;
		index_type wlnode_end;
		const int _NP_CROSSOVER_WP = 32;
		const int BLKSIZE = __kernel_tb_size;
		const int ITSIZE = BLKSIZE * 4;
		int* devF2Size = devSizes + 2;
		int nvwwarps = nthreads / vwSize;
		int vwid = tid / vwSize;
		int vwlaneid = tid % vwSize;
		int flag;

		typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
		typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

		__shared__ npsTy nps ;

		wlnode_end = roundup(devSizes[4], blockDim.x / vwSize);
		// int numofblock = wlnode_end / (blockDim.x / vwSize);
		// int bid = tid / __kernel_tb_size;
		// if(tid == 0){
		// 	printf("22221:%d,%d,%d\n",wlnode_end,tid,numofblock);
		// }
		for (index_type wlnode = vwid; wlnode < wlnode_end; wlnode += nvwwarps)
		{
			int node = -1;
			bool pop = false;
			int nodeS = 0,nodeE = 0;
			multiple_sum<2, index_type> _np_mps;
			multiple_sum<2, index_type> _np_mps_total;
			if(vwlaneid == 0 && wlnode < devSizes[4]){
				node = devF3[wlnode];
				nodeS = devNodes[node];
				nodeE = devNodes[node + 1];
				pop = ((nodeE - nodeS) < DEGREE_LIMIT) ? true: false;
			}
			struct NPInspector1 _np = {0,0,0,0,0,0};
			__shared__ struct { int node; } _np_closure [TB_SIZE];
			_np_closure[threadIdx.x].node = node;
			if (node >= 0)
			{
				_np.size = nodeE-nodeS;
				_np.start = nodeS;
			}
			// if(tid == 0){
			// 	printf("22222:%d,%d,%d,%d\n",tid,node,_np.size,_np.start);
			// }
			_np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
			_np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
			BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
			// if(tid < 64){
			// 	printf("22223:%d,%d,%d,%d\n",tid,_np.size,_np_mps.el[1],_np_mps_total.el[1]);
			// }
			if (threadIdx.x == 0)
			{
			}
			__syncthreads();
			{
				const int warpid = threadIdx.x / 32;
				const int _np_laneid = threadIdx.x % 32;
				while (__any_sync(0xffffffff, _np.size >= _NP_CROSSOVER_WP))
				{
					if (_np.size >= _NP_CROSSOVER_WP)
					{
						nps.warp.owner[warpid] = _np_laneid;
					}
					__syncwarp(0xffffffff);
					if (nps.warp.owner[warpid] == _np_laneid)
					{
						nps.warp.start[warpid] = _np.start;
						nps.warp.size[warpid] = _np.size;
						nps.warp.src[warpid] = threadIdx.x;
						_np.start = 0;
						_np.size = 0;
					}
					__syncwarp(0xffffffff);
					index_type _np_w_start = nps.warp.start[warpid];
					index_type _np_w_size = nps.warp.size[warpid];
					node = _np_closure[nps.warp.src[warpid]].node;
					// printf("22225:%d,%d,%d,%d\n",tid,node,_np_w_size,_np_w_start);
					for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
					{
						int2 dest = devEdges[_np_ii + _np_w_start];
						int newWeight = devDistances[node].y + dest.y;
						int2 toWrite = make_int2(level, newWeight);
						unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
							reinterpret_cast<unsigned long long &>(toWrite));
						int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
						flag = ((oldNode2Weight.y > newWeight) && (level > oldNode2Weight.x));							
						if(flag){
							int pos = atomicAdd(devF2Size,1);
							devF2[pos] = dest.x;
						}
					}
				}
				__syncthreads();
			}

			__syncthreads();
			_np.total = _np_mps_total.el[1];
			_np.offset = _np_mps.el[1];
			while (_np.work())
			{
				int _np_i =0;
				_np.inspect2(nps.fg.itvalue, nps.fg.src, ITSIZE, threadIdx.x);
				__syncthreads();

				for (_np_i = threadIdx.x; _np_i < ITSIZE && _np.valid(_np_i); _np_i += BLKSIZE)
				{
					index_type edge;
					node = _np_closure[nps.fg.src[_np_i]].node;
					edge= nps.fg.itvalue[_np_i];
					int2 dest = devEdges[edge];
					// printf("22224:%d,%d,%d,%d,%d\n",tid,_np.size,node,edge,dest.x);
					int newWeight = devDistances[node].y + dest.y;
					int2 toWrite = make_int2(level, newWeight);
					unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
						reinterpret_cast<unsigned long long &>(toWrite));
					int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
					flag = ((oldNode2Weight.y > newWeight) && (level > oldNode2Weight.x));
					if(flag){
						int pos = atomicAdd(devF2Size,1);
						devF2[pos] = dest.x;
					}
				}
				_np.execute_round_done(ITSIZE);
				__syncthreads();
			}
			node = _np_closure[threadIdx.x].node;
		}
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
		GlobalBarrier gb,
		int vwSize,
		int strategyNum,
		int clockFrec,
		float* times
	)
	{
		thread_block g = this_thread_block();
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int nthreads = gridDim.x * blockDim.x;

		const int warpSharedLimit2 = 32;
		__shared__ int sharedPtr[256*2];
		int *queue1 = sharedPtr + g.thread_rank() / WARPSIZE * warpSharedLimit2;
		int *queue2 = sharedPtr + 256 + g.thread_rank() / WARPSIZE * warpSharedLimit2;
	  
		float distanceLimit = initDL;

		int F1Index = 0;
		int F2index = 0;
		int F3index = 0;
		int level = 0;
		int relaxNodes = 0;
		unsigned long long tt1=0,tt2=0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;


		while(devSizes[F1Index] > 0){
			level = level +1;
			clock_t start=clock();
			SelectNodesV9(devDistances,devF1, devF2, devF3, devSizes,F1Index,F2index,F3index,
				distanceLimit,warpSharedLimit2,realID,tile,queue1,queue2,mymask,level);
			gb.Sync();
			clock_t t1 =clock() - start;
			// if(realID == 0){
			// 	printf("00000:%d,%d,%d,%d\n",devSizes[0],devSizes[2],devSizes[4],level);
			// }

			if(strategyNum == 0){ //BCE
				distanceLimit += dl;
			}else if(strategyNum > 0 && devSizes[4+F3index] < strategyNum){ // PBCE
				distanceLimit += dl;
			}else if(strategyNum == -1 && devSizes[0+F3index] == 0){ //delta
				distanceLimit += dl;
			}

			sssp_kernel_dev(realID,nthreads,devNodes,devEdges,devDistances,devF1,devF2,devF3,devSizes,level,vwSize);
			gb.Sync();
			clock_t t2 = clock() - start;
			// if(realID == 0){
			// 	printf("11111:%d,%d,%d,%d\n",devSizes[0],devSizes[2],devSizes[4],level);
			// }
			swapPtr(devF1,devF2);

			if(realID == 0){
				relaxNodes += devSizes[4 + F3index];
				devSizes[F1Index] = devSizes[2+F2index];
				devSizes[2+F2index] = 0;
				devSizes[4+F3index] = 0;
				tt1 += t1;
				tt2 += t2;
			}
			gb.Sync();
		}

		if(realID == 0){
			atomicAdd(devSizes+6,relaxNodes);
			atomicAdd(devSizes+7,level);
			times[0] = ((float)tt1)/clockFrec;
			times[1] = ((float)(tt2-tt1))/clockFrec;
			// printf("cac:%lf\n",(double)tt1 / clockFrec);
			// printf("select:%lf\n",((double)(tt2-tt1))/clockFrec);
		}
	}	
}

using namespace KernelV9;

#define kernelV9Atmoic64(gdim, bdim,sharedLimit,initDL,distanceLimit,gb) \
HBFSearchV9Atomic64<<<gdim, bdim >>> \
(devUpOutNodes, devUpOutEdges, devInt2Distances, devF1, devF2,devF3, devSizes, sharedLimit, initDL,distanceLimit,gb,vwSize,strategyNum,clockFrec,times)