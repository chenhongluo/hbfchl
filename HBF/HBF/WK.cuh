#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#define WARPSIZE 32
namespace WriteKernel
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
	__device__ __forceinline__ int
		SWrite(thread_block_tile<VW_SIZE>& tile,
			T* writeQueueAddr, int * writeSizeAddr,
			int flag, T data,
			int * queue, int &queueSize, const int queueLimit,
			unsigned &mymask)
	{
		int ret = 0;
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
			queueSize = 0;
			ret = 1;
		}
		tile.sync();
		if (flag)
		{
			// write to shared mem
			mask = mask & mymask;
			int pos = __popc(mask);
			queue[pos + queueSize] = data;
		}
		tile.sync();
		queueSize += sum;
		return ret;
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
	__global__ void WriteKernel4(
		int* devF3, int* devF2,
		int *__restrict__ devSizes,
		int nl,
		const int sharedLimit)
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
		const int warpSharedLimit = (sharedLimit / 4) / blockdim * WARPSIZE;
		int* devF2Size = devSizes + 1;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + g.thread_rank() / WARPSIZE * warpSharedLimit;
		int queueSize = 0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;
		int at = 0;

		int ttt = devSizes[2] - devSizes[2] % WARPSIZE;
		for (int i = tileID; i < ttt; i += IDStride)
		{
			int v = devF3[i];
			int flag = 0;
			if(v < nl)
			{
				flag = 1;
			}
			at += SWrite<WARPSIZE, int>(tile, devF2, devF2Size, flag, v, queue, queueSize, warpSharedLimit, mymask);
		}
		at += SWrite< WARPSIZE, int>(tile, devF2, devF2Size, 0, 0, queue, queueSize, 0, mymask);
		if (tile.thread_rank() == 0) {
			atomicAdd(devSizes + 3, at);
		}
	}

	template <int VW_SIZE>
	__global__ void WriteKernel3(
		int* devF3, int* devF2,
		int *__restrict__ devSizes,
		int nl,
		const int sharedLimit)
	{
		//alloc node&edge
		thread_block g = this_thread_block();
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int tileID = realID / VW_SIZE;
		const int IDStride = gridDim.x * (blockdim / VW_SIZE);
		const int threadLimit = (sharedLimit / 4) / blockdim;
		int* devF2Size = devSizes + 1;
		int at = 0;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + threadLimit * g.thread_rank();
		int founds = 0;

		int ttt = devSizes[2] - devSizes[2] % WARPSIZE;
		for (int i = tileID; i < devSizes[0]; i += IDStride)
		{
			int v = devF3[i];
			if(v < nl)
			{
				queue[founds++] = v;
			}
			if (tile.any(founds >= threadLimit)) {
				VWWrite<WARPSIZE, int>(tile, devF2Size, devF2, founds, queue);
				founds = 0;
				at++;
			}
		}
		// write to global mem
		VWWrite<WARPSIZE, int>(tile, devF2Size, devF2, founds, queue);
		at++;
		if (tile.thread_rank() == 0) {
			atomicAdd(devSizes + 3, at);
		}
	}	

	template <int VW_SIZE>
	__global__ void WriteKernel2(
		int* devF3, int* devF2,
		int *__restrict__ devSizes,
		int nl,
		const int sharedLimit)
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
		const int warpSharedLimit = (sharedLimit / 4) / blockdim * WARPSIZE;
		int* devF2Size = devSizes + 1;

		// for write in shared mem
		extern __shared__ int st[];
		int *queue = st + g.thread_rank() / WARPSIZE * warpSharedLimit;
		int queueSize = 0;
		unsigned mymask = (1 << tile.thread_rank()) - 1;
		int at = 0;

		int ttt = devSizes[2] - devSizes[2] % WARPSIZE;
		for (int i = tileID; i < ttt; i += IDStride)
		{
			int v = devF3[i];
			int flag = 0;
			if(v < nl)
			{
				flag = 1;
			}
			unsigned mask = tile.ballot(flag);
			int sum = __popc(mask);
			int globalBias;
			if (tile.thread_rank() == 0)
				globalBias = atomicAdd(devF2Size, sum);
			globalBias = tile.shfl(globalBias, 0);
			if (flag)
			{
				mask = mask & mymask;
				int pos = __popc(mask);
				devF2[pos] = v;
			}
			at++;
		}
		if (tile.thread_rank() == 0) {
			atomicAdd(devSizes + 3, at);
		}
	}

	template <int VW_SIZE>
	__global__ void WriteKernel1(
		int* devF3, int* devF2,
		int *__restrict__ devSizes,
		int nl,
		const int sharedLimit)
	{
		//alloc node&edge
		thread_block g = this_thread_block();
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		// dim3 group_index();
		// dim3 thread_index();
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int tileID = realID / VW_SIZE;
		const int IDStride = gridDim.x * (blockdim / VW_SIZE);
		const int threadLimit = (sharedLimit / 4) / blockdim;
		int* devF2Size = devSizes + 1;
		int at = 0;

		int ttt = devSizes[2] - devSizes[2] % WARPSIZE;
		for (int i = tileID; i < ttt; i += IDStride)
		{
			int v = devF3[i];
			int flag = 0;
			if(v < nl)
			{
				flag = 1;
			}
			tile.sync();
			if (flag)
			{
				int globalBias = atomicAdd(devF2Size, 1);
				devF2[globalBias] = v;
				at++;
			}
		}
		// write to global mem
		int sumat = 0;
		VWInclusiveScanAdd<WARPSIZE,int>(tile,at,sumat);
		if (tile.thread_rank() == WARPSIZE -1) {
			atomicAdd(devSizes + 3, sumat);
		}
	}
}

using namespace WriteKernel;

#define writeKernel(wc,gridDim, blockDim,sharedLimit) \
wc<1><<<gridDim, blockDim, sharedLimit>>> \
(devF3, devF2, devSizes, nl,sharedLimit);

#define switchWriteKernel(wc) \
	writeKernel(wc,gdim, bdim,sharedLimit)