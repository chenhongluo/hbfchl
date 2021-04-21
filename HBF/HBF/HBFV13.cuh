#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "gbar.cuh"
#include "WorkList.cuh"
using namespace cooperative_groups;

#define WARPSIZE 32
#define MAXLELVEL 0x0FFFFFFF
#define ABbit 13
namespace KernelV13
{
	__device__ void swapPtr(int* & a,int* & b){
		int * c = b;
		b = a;
		a = c;
	}

	__device__ void getLevel(int nl,int nq, int ssLevel, int ggLevel,int alevel,int blevel,int& qLevel){ 
		// BCE -> flag =false,minl = alevel+1, minq = alevel+1 -ssLevel; 
		// PBCE -> flag =true,minl = alevel, minq = sp
		if(nl <= alevel){
			nl = alevel;
			if(blevel > 0){
				qLevel = nq;
				return;
			}else{
				nl = nl +1;
			}
		}
		if(nl >= ggLevel){
			qLevel = ggLevel - ssLevel;
		}else{
			qLevel = nl - ssLevel;
		}
		// printf("%d, %d,%d\n",nl,indexLevel,qLevNel);
	}

	__device__ void SelectNodesV13(
		int *__restrict__ devDistances,
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
				if(1){
					int indexLevel;
					int qLevel;
					indexLevel = devDistances[index] / dl;
					if(indexLevel < alevel){
						indexLevel = alevel;
					}
					if(indexLevel >= ggLevel){
						indexLevel = ggLevel - 1;
					}
					qLevel = indexLevel - ssLevel;
					int pos = workLists.getPos(qLevel,1);
					workLists.setData(qLevel,pos,index);
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

	__device__ void sssp_kernel_dev_V13_vw(
		unsigned tid,unsigned nthreads,
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int *__restrict__ devDistances,
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
				sourceWeight = devDistances[index];
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
					int qLevel;
					getLevel(newWeight/dl, to, ssLevel, ggLevel, alevel, blevel, qLevel);
					// indexLevel = alevel + 1;
					// qLevel = indexLevel;

					// int oldWeight = atomicMin(&devDistances[dest.x],newWeight);
					// flag = (oldWeight > newWeight);
					// if(flag){
					// 	int pos = workLists.getPos(qLevel,1);
					// 	workLists.setData(qLevel,pos,dest.x);
					// }

					flag = (devDistances[dest.x] > newWeight);
					if(flag){
						atomicMin(&devDistances[dest.x],newWeight);
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
	HBFSearchV13Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int *__restrict__ devDistances,
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
		// unsigned long long tt1=0,tt2=0;
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
						sssp_kernel_dev_V13_vw(realID,nthreads,devNodes,devEdges,devDistances,
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
						sssp_kernel_dev_V13_vw(realID,nthreads,devNodes,devEdges,devDistances,
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
			// if(realID == 0){
			// 	printf("11111:%d,%d,%d,%d\n",alevel,mainq,workLists.getMaxSizeOfQueue(mainq));
			// }
			SelectNodesV13(devDistances,workLists,realID,dl,mainq,qnum,alevel);
			gb.Sync();
			if(realID == 0){
				workLists.clear(mainq);
			}
			gb.Sync();
		}

		if(relaxNodes > 0)
			atomicAdd(devSizes+6,relaxNodes);
		if(realID == 0){
			atomicAdd(devSizes+7,allLevel);
			// times[1] = ((float)tt1)/clockFrec;
			// times[0] = ((float)(tt2-tt1))/clockFrec;
			// printf("%ul\n",tt1);
			// printf("%ul\n",tt2);
		}
	}	
}

using namespace KernelV13;

#define kernelV13Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb) \
HBFSearchV13Atomic64<<<gdim, bdim >>> \
(devUpOutNodes, devUpOutEdges, devIntDistances,devSizes, workLists, qnum, distanceLimit,gb,strategyNum,vwSize,clockFrec,times)