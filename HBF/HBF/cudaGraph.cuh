#pragma once
#include "graph.h"
#include <climits>
#include <condition_variable>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::cerr << std::endl << " CUDA error "
			<< StreamModifier::Emph::SET_UNDERLINE << file
			<< "(" << line << ")"
			<< StreamModifier::Emph::SET_RESET << " : " << errorMessage
			<< " -> " << cudaGetErrorString(err) << "(" << (int)err
			<< ") " << std::endl << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
}

#define __CUDA_ERROR(msg)                                                       \
                    {                                                           \
                        cudaDeviceSynchronize();                                \
                        __getLastCudaError (msg, __FILE__, __LINE__);			\
                    }

using namespace graph;
namespace cuda_graph {
	struct ValidStruct{
		int d;
		int flag;
	};

	class CudaConfigs {
	public:
		string kernelVersion;
		bool atomic64;
		int vwSize;
		int gridDim;
		int blockDim;
		int sharedLimit;
		int bcea;
		int bcek;
		string distanceLimitStrategy;
		float distanceLimit;
		int dp;
		CudaConfigs() {
			atomic64 = true;
			dp = INT_MAX;
		}
		CudaConfigs(string kv,int vs,int gd,int bd,int sl,string dls,float dl){
			kernelVersion = kv;
			atomic64 = true;
			vwSize = vs;
			gridDim = gd;
			blockDim = bd;
			sharedLimit = sl;
			distanceLimitStrategy = dls;
			distanceLimit = dl;
			dp = INT_MAX;
			if(dls.find("strict")==0?1:0){
				vector<string> sss = stringUtil::split(dls, "-");
				distanceLimitStrategy = "strict";
				bcea = atoi(sss[1].c_str());
				bcek = atoi(sss[2].c_str());
			}
		}
	};
	class CudaProfiles {
	public:
		long relaxNodes;
		long relaxEdges;
		long relaxRemain;
		double kernel_time, copy_time, select_time, cac_time;
		int depth;
		int v, e;

		CudaProfiles() {
			relaxNodes = relaxEdges = relaxRemain = 0;
			kernel_time = cac_time = copy_time = select_time = 0.0;
			depth = 0;
		}

		void cac(){
			cac_time = kernel_time - select_time;
		}
	};
	class CudaGraph : ComputeGraph {
	private:
		int v, e;
	public:
		GraphWeight &gp;
		int* f1, *f2, *f3;
		int *devSizes;
		int *devMM;

		int *devUpOutNodes;
		int2 *devUpOutEdges;

		int *devIntDistances;
		int2 *devInt2Distances;

		int2 *devTrueInt2Distances;
		int2 *validRes;
		int* validSizes;
		int2 *devBF;
		int* devBFSize;

		int* devPF1,*devPF2,*devPF3;
		int* devPFSize1,*devPFSize2,*devPFSize3;

		CudaConfigs configs;
		CudaGraph(GraphWeight & _gp, CudaConfigs  _configs);
		float nodeAllocTest(vector<int> sources,int n, CudaProfiles & profile);
		float nodeWriteTest(vector<int> sources,int n,int nl, CudaProfiles & profile);
		void* computeAndTick(node_t source, vector<dist_t>& res, double & t);
		void cacValid(node_t source,int printInterval);
		~CudaGraph();
	private:
		void cudaMallocMem();
		void cudaFreeMem();
		void cudaCopyMem();
		void cudaInitComputer(int initNode);
		void cudaGetRes(vector<int> &res);
		void searchV0(int source, CudaProfiles& profile);
		void searchV1(int source, CudaProfiles& profile);
		void searchV2(int source, CudaProfiles& profile);
		void initValidRes();
		void getValidRes(int level,int printInterval,float distanceLimit);
		int getTrueDistance(int source);
	};
	void justTest();
	/*class CHCudaGraph : ComputeGraph {
	private:
		CudaGraph *upGraph;
		CudaGraph *downGraph;
	public:
		void search(int source, CudaProfiles& profile);
	};*/
}