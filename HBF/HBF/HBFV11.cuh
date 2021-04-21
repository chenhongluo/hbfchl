#include <ctime>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "gbar.cuh"
#include "WorkList.cuh"
using namespace cooperative_groups;

#define WARPSIZE 32
#define MAXLELVEL 1000000
#define ABbit 20
// if write lisan ,turn false else turn true
namespace KernelV11
{
	__device__ void swapPtr(int* & a,int* & b){
		int * c = b;
		b = a;
		a = c;
	}

	__device__ __forceinline__ void getLevel(int distance,float dl,int ssLevel,int ggLevel,int alevel,int &indexLevel,int& qLevel){
		indexLevel = distance / dl;
		if(indexLevel <= alevel){
			indexLevel = alevel + 1;
		}
		if(indexLevel >= ggLevel){
			indexLevel = MAXLELVEL;
			qLevel = ggLevel - ssLevel;
		}else{
			qLevel = indexLevel - ssLevel;
		}
		// indexLevel = alevel + 1;
		// qLevel = indexLevel - ssLevel;
	}

	__device__ void SelectNodesV11(
		int2 *__restrict__ devDistances,
		WorkListsPlain workLists,
		int realID,
		const float dl,
		int mainq,
		int qnum,
		int alevel)
	{
		const int warpID = realID / WARPSIZE;
		const int laneId = realID % WARPSIZE;
		const int warpStride = gridDim.x * (blockDim.x / WARPSIZE);
		int devMaxSize = workLists.getMaxSizeOfQueue(mainq);
		int ssLevel = alevel - alevel%qnum;
		int ggLevel = ssLevel + qnum;
		for (int i = warpID; i < (devMaxSize + WARPSIZE - 1) / WARPSIZE; i += warpStride)
		{
			int j = i * WARPSIZE + laneId;
			int index = -1;
			if (j < devMaxSize) {
				index = workLists.getData(qnum,j);
				int oldLevel = devDistances[index].x;
				if(oldLevel == MAXLELVEL){
				// if(1){
					int indexLevel;
					int qLevel;
					indexLevel = devDistances[index].y / dl;
					if(indexLevel < alevel){
						indexLevel = alevel;
					}
					if(indexLevel >= ggLevel){
						indexLevel = ggLevel - 1;
					}
					qLevel = indexLevel - ssLevel;
					int pos = workLists.getPos(qLevel,1);
					workLists.setData(qLevel,pos,index);
					devDistances[index].x = indexLevel;
				}
			}
		}
	}
	__device__ int roundup(int data, int mult) {
		auto rem = data % mult;
	  
		if (!rem)
		  return data;
		return data + (mult - rem);
	  }

	__device__ __forceinline__ void sssp_kernel_dev_V11_vw(
		unsigned realID,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int *__restrict__ devSizes,
		int2 *__restrict__ devDistances,
		int2 *__restrict__ devTrueDistances,
		WorkListsPlain workLists,
		const float dl,
		int mainq,
		int qnum,
		int alevel,int blevel,
		int vwSize,int& relaxNodes
	){
		int vwn = WARPSIZE/vwSize;
		const int tileID = realID / vwSize;
		const int tileThreadId = realID % vwSize;
		const int IDStride = gridDim.x * (blockDim.x / vwSize);
		int ssLevel = alevel - alevel%qnum;
		int ggLevel = ssLevel + qnum;
		int iterations = workLists.getMaxSizeOfQueue(mainq);
		int ta = 0,tb = 0;
		if(iterations % vwn != 0)
			iterations = iterations + (vwn - iterations % vwn);
		for (int i = tileID; i < iterations; i += IDStride)
		{
			int index,sourceWeight,nodeS = 0,nodeE = 0,relaxSize = 0,maxRelaxSize = 0;
			if(i < workLists.getMaxSizeOfQueue(mainq)){
				index = workLists.getData(mainq,i);
				sourceWeight = devDistances[index].y;
				if(devDistances[index].x != alevel){
					continue;
				}
				nodeS = devNodes[index];
				nodeE = devNodes[index + 1];
				relaxSize = nodeE - nodeS;
				maxRelaxSize = 0;
			}
			if(tileThreadId == 0){
				relaxNodes++;
			}

			#if Profile
			if(tileThreadId == 0 && i < workLists.getMaxSizeOfQueue(mainq)){
				atomicAdd(devSizes + 11,1);
				// ta ++;
				if(sourceWeight == devTrueDistances[index].y){
					atomicAdd(devSizes + 12,1);
					// tb++;
				}
			}
			#endif
			// if(ta > 1){
			// 	atomicAdd(devSizes+32,ta);
			// 	atomicAdd(devSizes+64,ta);
			// 	ta = 0;
			// 	tb = 0;
			// }
			// VWInclusiveScanMax<WARPSIZE,int>(tile,relaxSize,maxRelaxSize);
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
					int indexLevel,qLevel;
					getLevel(newWeight,dl,ssLevel,ggLevel,alevel,indexLevel,qLevel);
					// indexLevel = alevel + 1;
					// qLevel = indexLevel;
					int2 toWrite = make_int2(indexLevel, newWeight);
					if(newWeight < devDistances[dest.x].y){
						unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
							reinterpret_cast<unsigned long long &>(toWrite));
						int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
						flag = ((oldNode2Weight.y > newWeight) && (indexLevel != oldNode2Weight.x));
						if(flag){
							int pos = workLists.getPos(qLevel,1);
							workLists.setData(qLevel,pos,dest.x);
						}
					}
					// int compLevel = devDistances[dest.x].x;
					// 	if(newWeight < devDistances[dest.x].y){
					// 		atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
					// 			reinterpret_cast<unsigned long long &>(toWrite));
					// 		flag = indexLevel != compLevel;
					// 		if(flag){
					// 			int pos = workLists.getPos(qLevel,1);
					// 			workLists.setData(qLevel,pos,dest.x);
					// 		}
					// 	}
				}
			}
		}
		// if(ta > 0){
		// 	atomicAdd(devSizes+11,ta);
		// 	atomicAdd(devSizes+12,ta);
		// 	ta = 0;
		// 	tb = 0;
		// }
	}

	__device__ void sssp_kernel_dev_V11_a(
		unsigned tid,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		WorkListsPlain workLists,
		const float dl,
		int mainq,
		int qnum,
		int alevel,int blevel,
		int vwSize,int &relaxNodes
	)
	{
		const unsigned __kernel_tb_size = __tb_sssp_kernel;
		typedef int index_type;
		index_type wlnode_end;
		const int _NP_CROSSOVER_WP = INT_MAX;
		const int BLKSIZE = __kernel_tb_size;
		const int ITSIZE = BLKSIZE * 2;
		int nvwwarps = nthreads / vwSize;
		int vwid = tid / vwSize;
		int vwlaneid = tid % vwSize;
		int ssLevel = alevel - alevel%qnum;
		int ggLevel = ssLevel + qnum;
		// int relaxEdges = 0;

		typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
		typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

		__shared__ npsTy nps ;

		wlnode_end = roundup(workLists.getMaxSizeOfQueue(mainq), blockDim.x / vwSize);
		for (index_type wlnode = vwid; wlnode < wlnode_end; wlnode += nvwwarps)
		{
			int node = -1;
			bool pop = false;
			int nodeS = 0,nodeE = 0;
			multiple_sum<2, index_type> _np_mps;
			multiple_sum<2, index_type> _np_mps_total;
			if(vwlaneid == 0 && wlnode < workLists.getMaxSizeOfQueue(mainq)){
				node = workLists.getData(mainq,wlnode);
				if(devDistances[node].x == alevel){
				// if(1){
					nodeS = devNodes[node];
					nodeE = devNodes[node + 1];
					pop = ((nodeE - nodeS) < DEGREE_LIMIT) ? true: false;
					relaxNodes++;
				}else{
					node = -1;
				}
			}
			struct NPInspector1 _np = {0,0,0,0,0,0};
			__shared__ struct { int node; } _np_closure [TB_SIZE];
			_np_closure[threadIdx.x].node = node;
			if (node >= 0)
			{
				_np.size = nodeE-nodeS;
				_np.start = nodeS;
			}
			_np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
			_np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
			BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);
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
					for (int _np_ii = _np_laneid; _np_ii < _np_w_size; _np_ii += 32)
					{
						int2 dest = devEdges[_np_ii + _np_w_start];
						int newWeight = devDistances[node].y + dest.y;
						int indexLevel,qLevel;
						getLevel(newWeight,dl,ssLevel,ggLevel,alevel,indexLevel,qLevel);
						int2 toWrite = make_int2(indexLevel, newWeight);
						unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
							reinterpret_cast<unsigned long long &>(toWrite));
						int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
						int flag = ((oldNode2Weight.y > newWeight) && (indexLevel != oldNode2Weight.x));
						// int flag = 	(oldNode2Weight.y > newWeight);						
						if(flag){
							int pos = workLists.getPos(qLevel,1);
							workLists.setData(qLevel,pos,dest.x);
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
					// relaxEdges ++;
					index_type edge;
					node = _np_closure[nps.fg.src[_np_i]].node;
					edge= nps.fg.itvalue[_np_i];
					int2 dest = devEdges[edge];
					int newWeight = devDistances[node].y + dest.y;
					int indexLevel,qLevel;
					getLevel(newWeight,dl,ssLevel,ggLevel,alevel,indexLevel,qLevel);
					int2 toWrite = make_int2(indexLevel, newWeight);
					if(newWeight < devDistances[dest.x].y){
						unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
							reinterpret_cast<unsigned long long &>(toWrite));
						int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
						int flag = ((oldNode2Weight.y > newWeight) && (indexLevel != oldNode2Weight.x));
						// int flag = 	(oldNode2Weight.y > newWeight);						
						if(flag){
							int pos = workLists.getPos(qLevel,1);
							workLists.setData(qLevel,pos,dest.x);
						}
					}
				}
				_np.execute_round_done(ITSIZE);
				__syncthreads();
			}
			node = _np_closure[threadIdx.x].node;
		}
		// if(threadIdx.x == 0){
		// 	printf("%d,%d,%d\n",alevel,blockIdx.x,relaxEdges);
		// }
	}

	__device__ void sssp_kernel_dev_V11_b(
		unsigned tid,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		WorkListsPlain workLists,
		const float dl,
		int mainq,
		int qnum,
		int alevel,int blevel,
		int oneSize, int* readSize, int &relaxNodes
	)
	{
		const unsigned __kernel_tb_size = __tb_sssp_kernel;
		typedef int index_type;
		index_type wlnode_end;
		const int _NP_CROSSOVER_WP = INT_MAX;
		const int BLKSIZE = __kernel_tb_size;
		const int ITSIZE = BLKSIZE * 2;
		// int nvwwarps = nthreads / vwSize;
		// int vwid = tid / vwSize;
		// int vwlaneid = tid % vwSize;
		int vsize = blockDim.x / oneSize;
		__shared__ int blockOneSize[1]; 
		int ssLevel = alevel - alevel%qnum;
		int ggLevel = ssLevel + qnum;
		// int relaxEdges = 0;

		typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
		typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

		__shared__ npsTy nps ;

		while(1)
		{
			__syncthreads();
			if(threadIdx.x == 0){
				blockOneSize[0] = atomicAdd(readSize,oneSize);
			}
			__syncthreads();

			if(blockOneSize[0] > workLists.getMaxSizeOfQueue(mainq)){
				// printf("11111:%d,%d,%d,%d\n",blockIdx.x,threadIdx.x,blockOneSize[0],workLists.getMaxSizeOfQueue(mainq));
				return;
			}
			// printf("22222:%d,%d,%d,%d\n",blockIdx.x,threadIdx.x,blockOneSize[0],workLists.getMaxSizeOfQueue(mainq));

			// if(threadIdx.x == 255){
			// 	printf("22222:%d,%d,%d,%d\n",blockIdx.x,threadIdx.x,blockOneSize[0],workLists.getMaxSizeOfQueue(mainq));
			// }
			int wlnode = blockOneSize[0] + threadIdx.x / vsize;
			int node = -1;
			bool pop = false;
			int nodeS = 0,nodeE = 0;
			multiple_sum<2, index_type> _np_mps;
			multiple_sum<2, index_type> _np_mps_total;
			if(threadIdx.x % vsize == 0 && wlnode < workLists.getMaxSizeOfQueue(mainq)){
				node = workLists.getData(mainq,wlnode);
				if(devDistances[node].x == alevel){
				// if(1){
					nodeS = devNodes[node];
					nodeE = devNodes[node + 1];
					pop = ((nodeE - nodeS) < DEGREE_LIMIT) ? true: false;
					relaxNodes+=1;
				}else{
					node = -1;
				}
			}
			struct NPInspector1 _np = {0,0,0,0,0,0};
			__shared__ struct { int node; } _np_closure [TB_SIZE];
			_np_closure[threadIdx.x].node = node;
			if (node >= 0)
			{
				_np.size = nodeE-nodeS;
				_np.start = nodeS;
			}
			_np_mps.el[0] = _np.size >= _NP_CROSSOVER_WP ? _np.size : 0;
			_np_mps.el[1] = _np.size < _NP_CROSSOVER_WP ? _np.size : 0;
			BlockScan(nps.temp_storage).ExclusiveSum(_np_mps, _np_mps, _np_mps_total);

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
					// relaxEdges ++;
					index_type edge;
					node = _np_closure[nps.fg.src[_np_i]].node;
					edge= nps.fg.itvalue[_np_i];
					int2 dest = devEdges[edge];
					int newWeight = devDistances[node].y + dest.y;
					int indexLevel,qLevel;
					if(newWeight < devDistances[dest.x].y){
						getLevel(newWeight,dl,ssLevel,ggLevel,alevel,indexLevel,qLevel);
						int2 toWrite = make_int2(indexLevel, newWeight);
						unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
							reinterpret_cast<unsigned long long &>(toWrite));
						int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
						int flag = ((oldNode2Weight.y > newWeight) && (indexLevel != oldNode2Weight.x));
						// int flag = 	(oldNode2Weight.y > newWeight);						
						if(flag){
							int pos = workLists.getPos(qLevel,1);
							workLists.setData(qLevel,pos,dest.x);
						}
					}
				}
				_np.execute_round_done(ITSIZE);
				__syncthreads();
			}
			// if(threadIdx.x == 255 || threadIdx.x == 0){
			// 	printf("33333:%d,%d,%d,%d\n",blockIdx.x,threadIdx.x,blockOneSize[0],workLists.getMaxSizeOfQueue(mainq));
			// }
			node = _np_closure[threadIdx.x].node;
		}

		// if(threadIdx.x == 0){
		// 	printf("%d,%d,%d\n",alevel,blockIdx.x,relaxEdges);
		// }
	}

	__global__ void printDetail(int* devSizes,int clockFrec,unsigned long long maxCounts){
		clock_t start = clock();
		int prev1 = 0;//all
		int prev2 = 0;//valid
		clock_t tc = start;
		int i = 0;
		float pret = 0;
		float t = 0;
		if(blockIdx.x == 0 && threadIdx.x == 0){
			while((tc-start) < maxCounts){
				i = 0;
				// while(i<10000){
					// i++;
				// }
				float t = float(tc - start) / clockFrec;
				int now1 = *(int volatile*)(devSizes+11);
				int now2 = *(int volatile*)(devSizes+12);
	
				if(now1 > prev1 && t>pret+1){
					printf("%f %d %d %d %d\n",t,now1,now1-prev1,now2,now2 - prev2);

					prev1 = now1;
					prev2 = now2;
					pret = t;
				}
				tc = clock();
			}
		}
	}

	__global__ void __launch_bounds__(256) HBFSearchV11Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int2 *__restrict__ devTrueDistances,
		int* devSizes,
		WorkListsPlain workLists,
		int qnum,
		float dl,
		GlobalBarrier gb,
		int vwSize,
		int clockFrec,
		float* times,
		int kernelFlag
	)
	{
		thread_block g = this_thread_block();
		thread_block_tile<WARPSIZE> tile = tiled_partition<WARPSIZE>(g);
		const int blockdim = g.group_dim().x;
		const int realID = g.group_index().x * blockdim + g.thread_rank();
		const int nthreads = gridDim.x * blockDim.x;

		int alevel = 0,blevel = 0,allLevel = 0;
		int relaxNodes = 0;
		unsigned long long tt1=0,tt2=0;
		const int tileID = realID / vwSize;
		const int tileThreadId = realID % vwSize;
		const int IDStride = gridDim.x * (blockdim / vwSize);
		int* readSize = devSizes + 33;
		while(1){
			int mainq = 0;
			clock_t start = clock();
			while (mainq < qnum){
				// if(blockIdx.x == 68*2-1 && threadIdx.x == 0){
				// 	printf("00000:%d,%d,%d,%d\n",alevel,mainq,workLists.getMaxSizeOfQueue(mainq),*readSize);
				// }
				// gb.Sync();
				if(workLists.getMaxSizeOfQueue(mainq) > 0){
					if(kernelFlag == 0){
						sssp_kernel_dev_V11_vw(realID,nthreads,devNodes,devEdges,devSizes,devDistances,devTrueDistances,
							workLists,dl,mainq,qnum,alevel,blevel,vwSize,relaxNodes);
					}
					else if(kernelFlag == 1){
						sssp_kernel_dev_V11_a(realID,nthreads,devNodes,devEdges,devDistances,
							workLists,dl,mainq,qnum,alevel,blevel,vwSize,relaxNodes);
					}
					else if(kernelFlag == 2){
						sssp_kernel_dev_V11_b(realID,nthreads,devNodes,devEdges,devDistances,
							workLists,dl,mainq,qnum,alevel,blevel,vwSize,readSize,relaxNodes);
						// __syncthreads();
						// if(threadIdx.x == 0){
						// 	printf("55555:%d,%d,%d,%d\n",blockIdx.x,threadIdx.x);
						// }
						// gb.Sync();
						// sssp_kernel_dev_V11_vw_vw(realID,nthreads,devNodes,devEdges,devDistances,
						// 		workLists,dl,mainq,qnum,alevel,blevel,128,256,relaxNodes);
						// gb.Sync();
						// sssp_kernel_dev_V11_vw_vw(realID,nthreads,devNodes,devEdges,devDistances,
						// 		workLists,dl,mainq,qnum,alevel,blevel,256,INT_MAX,relaxNodes);
					}
					gb.Sync();
					if(realID == 0){
						workLists.clear(mainq);
						*readSize = 0;
					}
					if(readSize == devSizes + 33){
						readSize = devSizes + 34;
					}else{
						readSize = devSizes + 33;
					}
				}
				alevel ++;
				mainq ++;
				allLevel ++;
			}
			if(workLists.getMaxSizeOfQueue(mainq) == 0){
				break;
			}
			// if(realID == 0){
			// 	printf("11111:%d,%d,%d,%d\n",alevel,mainq,workLists.getMaxSizeOfQueue(mainq));
			// }
			clock_t t1 = clock() - start;
			SelectNodesV11(devDistances,workLists,realID,dl,mainq,qnum,alevel);
			if(realID == 0){
				workLists.clear(mainq);
			}
			gb.Sync();
			clock_t t2 = clock() - start;
			if(realID == 0){
				tt1 += t1;
				tt2 += t2;
			}
		}

		if(relaxNodes > 0)
			atomicAdd(devSizes+6,relaxNodes);
		if(realID == 0){
			atomicAdd(devSizes+7,allLevel);
			// atomicAdd(devSizes+8,tt1/clockFrec);
			// atomicAdd(devSizes+9,(tt2-tt1)/clockFrec);
			times[1] = ((float)tt1)/clockFrec;
			times[0] = ((float)(tt2-tt1))/clockFrec;
		}
	}	
}

using namespace KernelV11;

#if Profile
#define kernelV11Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb) \
HBFSearchV11Atomic64<<<gdim, bdim ,0,stream2>>> \
(devUpOutNodes, devUpOutEdges, devInt2Distances,devTrueInt2Distances,devSizes, workLists, qnum, distanceLimit,gb,vwSize,clockFrec,times,kernelFlag)
#else
#define kernelV11Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb) \
HBFSearchV11Atomic64<<<gdim, bdim>>> \
(devUpOutNodes, devUpOutEdges, devInt2Distances,devTrueInt2Distances,devSizes, workLists, qnum, distanceLimit,gb,vwSize,clockFrec,times,kernelFlag)
#endif