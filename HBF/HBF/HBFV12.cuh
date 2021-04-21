#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "gbar.cuh"
#include "WorkList.cuh"
using namespace cooperative_groups;

#define WARPSIZE 32
#define MAXLELVEL 0x0FFFFFFF
#define ABbit 13
namespace KernelV12
{
	__device__ void swapPtr(int* & a,int* & b){
		int * c = b;
		b = a;
		a = c;
	}

	__device__ void getLevel(int nl,int nq, int ssLevel, int ggLevel,int alevel,int blevel ,
		int &indexLevel,int& qLevel){ 
		// BCE -> flag =false,minl = alevel+1, minq = alevel+1 -ssLevel; 
		// PBCE -> flag =true,minl = alevel, minq = sp
		if(nl <= alevel){
			nl = alevel;
			if(blevel > 0){
				qLevel = nq;
				indexLevel = (alevel << ABbit) + blevel;
				return;
			}else{
				nl = nl +1;
			}
		}
		if(nl >= ggLevel){
			indexLevel = MAXLELVEL;
			qLevel = ggLevel - ssLevel;
		}else{
			indexLevel = nl << ABbit;
			qLevel = nl - ssLevel;
		}
		// printf("%d, %d,%d\n",nl,indexLevel,qLevNel);
	}

	__device__ void SelectNodesV12(
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
					devDistances[index].x = (indexLevel<<ABbit);
					// printf("33333:%d,%d,%d,%d,%d\n",MAXLELVEL,oldLevel,oldLevel >> ABbit,indexLevel,alevel);
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

	__device__ void sssp_kernel_dev_V12_a(
		unsigned tid,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		WorkListsPlain workLists,
		const float dl,
		int from,// workLists[from]
		int to,// >=0 -> PBCE write to workLists[to]; > -1 -> BCE
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
		const int ITSIZE = BLKSIZE * 4;
		int nvwwarps = nthreads / vwSize;
		int vwid = tid / vwSize;
		int vwlaneid = tid % vwSize;
		int ssLevel = alevel - alevel%qnum;
		int ggLevel = ssLevel + qnum;
		// int relaxEdges = 0;

		typedef cub::BlockScan<multiple_sum<2, index_type>, BLKSIZE> BlockScan;
		typedef union np_shared<BlockScan::TempStorage, index_type, struct empty_np, struct warp_np<__kernel_tb_size/32>, struct fg_np<ITSIZE> > npsTy;

		__shared__ npsTy nps ;

		wlnode_end = roundup(workLists.getMaxSizeOfQueue(from), blockDim.x / vwSize);
		for (index_type wlnode = vwid; wlnode < wlnode_end; wlnode += nvwwarps)
		{
			int node = -1;
			bool pop = false;
			int nodeS = 0,nodeE = 0;
			multiple_sum<2, index_type> _np_mps;
			multiple_sum<2, index_type> _np_mps_total;
			if(vwlaneid == 0 && wlnode < workLists.getMaxSizeOfQueue(from)){
				node = workLists.getData(from,wlnode);
				if((devDistances[node].x >> ABbit) == alevel){
				// if(1){
					nodeS = devNodes[node];
					nodeE = devNodes[node + 1];
					pop = ((nodeE - nodeS) < DEGREE_LIMIT) ? true: false;
					relaxNodes++;
				}else{
					// printf("11111:%d,%d,%d,%d\n",alevel,blevel,devDistances[node].x >> ABbit);	
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
					getLevel(newWeight/dl, to, ssLevel, ggLevel, alevel, blevel, indexLevel, qLevel);
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
				_np.execute_round_done(ITSIZE);
				__syncthreads();
			}
			node = _np_closure[threadIdx.x].node;
		}
		// if(threadIdx.x == 0){
		// 	printf("%d,%d,%d\n",alevel,blockIdx.x,relaxEdges);
		// }
	}

	__device__ void sssp_kernel_dev_V12_vw(
		unsigned tid,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int *__restrict__ devSizes,
		int2 *__restrict__ devDistances,
		int2 *__restrict__ devTrueDistances,
		WorkListsPlain workLists,
		const float dl,
		int from,// workLists[from]
		int to,// >=0 -> PBCE write to workLists[to]; > -1 -> BCE
		int qnum,
		int alevel,int blevel,
		int vwSize,int &relaxNodes
	)
	{
		int vwn = WARPSIZE/vwSize;
		const int tileID = tid / vwSize;
		const int tileThreadId = tid % vwSize;
		const int IDStride = gridDim.x * (blockDim.x / vwSize);
		int ssLevel = alevel - alevel%qnum;
		int ggLevel = ssLevel + qnum;
		int iterations = workLists.getMaxSizeOfQueue(from);
		if(iterations % vwn != 0)
			iterations = iterations + (vwn - iterations % vwn);
		for (int i = tileID; i < iterations; i += IDStride)
		{
			int index,sourceWeight,nodeS = 0,nodeE = 0,relaxSize = 0,maxRelaxSize = 0;
			if(i < workLists.getMaxSizeOfQueue(from)){
				index = workLists.getData(from,i);
				sourceWeight = devDistances[index].y;
				// if((devDistances[index].x >> ABbit) != alevel){
				// 	continue;
				// }
				nodeS = devNodes[index];
				nodeE = devNodes[index + 1];
				relaxSize = nodeE - nodeS;
				maxRelaxSize = 0;
			}
			if(tileThreadId == 0){
				relaxNodes++;
			}
			#if Profile
			if(tileThreadId == 0 && i < workLists.getMaxSizeOfQueue(from)){
				atomicAdd(devSizes + 11,1);
				// ta ++;
				if(sourceWeight == devTrueDistances[index].y){
					atomicAdd(devSizes + 12,1);
					// tb++;
				}
			}
			#endif
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
					getLevel(newWeight/dl, to, ssLevel, ggLevel, alevel, blevel, indexLevel, qLevel);
					// indexLevel = alevel + 1;
					// qLevel = indexLevel;
					int2 toWrite = make_int2(indexLevel, newWeight);
					unsigned long long aa = atomicMin(reinterpret_cast<unsigned long long *>(&devDistances[dest.x]),
						reinterpret_cast<unsigned long long &>(toWrite));
					int2 &oldNode2Weight = reinterpret_cast<int2 &>(aa);
					flag = ((oldNode2Weight.y > newWeight) && (indexLevel != oldNode2Weight.x));
					// flag = (oldNode2Weight.y > newWeight);
					if(flag){
						int pos = workLists.getPos(qLevel,1);
						workLists.setData(qLevel,pos,dest.x);
					}
				}
			}
		}
		// if(threadIdx.x == 0){
		// 	printf("%d,%d,%d\n",alevel,blockIdx.x,relaxEdges);
		// }
	}

	__global__ void
	HBFSearchV12Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int2 *__restrict__ devTrueDistances,

		int* devSizes,
		WorkListsPlain workLists,
		int qnum,
		float dl,
		GlobalBarrier gb,
		int strategyNum,
		int vwSize,
		int clockFrec,
		float* times
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
		// if(realID == 0){
		// 	int test = 1;
		// 	int test3 = test << 3;
		// 	int test33 = test3 >> 3;
		// 	printf("test:%d,%d,%d\n",test,test3,test33);
		// }
		// strategyNum = INT_MAX;

		while(1){
			int mainq = 0;
			int xq = mainq;
			clock_t start = clock();
			while (mainq < qnum){
				if(realID == 0){
					int ssLevel = alevel - alevel%qnum;
					int ggLevel = ssLevel + qnum;
					// printf("00000:a%d,b%d,%d,%d,%d\n",alevel,blevel,xq,workLists.getMaxSizeOfQueue(mainq),workLists.getMaxSizeOfQueue(33));
					// for(int kk = 0; kk < 32;kk++){
					// 	printf(",%d",workLists.getMaxSizeOfQueue(kk));
					// }
					// printf("\n");
				}
				// gb.Sync();
				if(workLists.getMaxSizeOfQueue(xq) > 0){
					if(workLists.getMaxSizeOfQueue(xq) < strategyNum){
						blevel = 0;
						sssp_kernel_dev_V12_vw(realID,nthreads,devNodes,devEdges,devSizes,devDistances,devTrueDistances,
							workLists,dl,xq, qnum+1,qnum,alevel,blevel,vwSize,relaxNodes);
						gb.Sync();
						if(realID == 0){
							workLists.clear(xq);
						}
						alevel ++;
						mainq ++;
						allLevel ++;
						xq = mainq;
					}else{
						blevel++;
						int xqq;
						if(xq == qnum+1){
							xqq = mainq;
						}else{
							xqq = qnum + 1;
						}
						sssp_kernel_dev_V12_vw(realID,nthreads,devNodes,devEdges,devSizes,devDistances,devTrueDistances,
							workLists, dl, xq, xqq, qnum, alevel, blevel, vwSize, relaxNodes);
						gb.Sync();
						if(realID == 0){
							workLists.clear(xq);
						}
						xq = xqq;
						allLevel ++;
						gb.Sync();
					}
				}else{
					alevel++;
					mainq++;
					xq = mainq;
				}
			}
			if(workLists.getMaxSizeOfQueue(mainq) == 0){
				break;
			}
			clock_t t1 = clock() - start;
			// if(realID == 0){
			// 	printf("11111:%d,%d,%d,%d\n",alevel,mainq,workLists.getMaxSizeOfQueue(mainq));
			// }
			SelectNodesV12(devDistances,workLists,realID,dl,mainq,qnum,alevel);
			gb.Sync();
			clock_t t2 = clock() - start;
			if(realID == 0){
				tt1 += t1;
				tt2 += t2;
			}
			if(realID == 0){
				workLists.clear(mainq);
			}
			gb.Sync();
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

using namespace KernelV12;

#if Profile
#define kernelV12Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb) \
HBFSearchV12Atomic64<<<gdim, bdim, 0,stream2>>> \
(devUpOutNodes, devUpOutEdges, devInt2Distances,devTrueInt2Distances,devSizes, workLists, qnum, distanceLimit,gb,strategyNum,vwSize,clockFrec,times)
#else
#define kernelV12Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb) \
HBFSearchV12Atomic64<<<gdim, bdim >>> \
(devUpOutNodes, devUpOutEdges, devInt2Distances,devTrueInt2Distances,devSizes, workLists, qnum, distanceLimit,gb,strategyNum,vwSize,clockFrec,times)
#endif