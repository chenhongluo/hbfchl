#pragma once
#include "graph.h"
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
	class CudaConfigs {
	public:
		string kernelVersion;
		bool atomic64;
		int vwSize;
		int gridDim;
		int blockDim;
		int sharedLimit;
		string distanceLimitStrategy;
		float distanceLimit;
		CudaConfigs() {
			atomic64 = true;
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
		GraphWeight &gp;
	public:
		int* f1, *f2, *f3;
		int *devSizes;
		int *devMM;

		int *devUpOutNodes;
		int2 *devUpOutEdges;

		int *devIntDistances;
		int2 *devInt2Distances;

		CudaConfigs configs;
		CudaGraph(GraphWeight & _gp, CudaConfigs  _configs);
		float nodeAllocTest(vector<int> sources,int n, CudaProfiles & profile);
		void* computeAndTick(node_t source, vector<dist_t>& res, double & t);
		~CudaGraph();
	private:
		void cudaMallocMem();
		void cudaFreeMem();
		void cudaCopyMem();
		void cudaInitComputer(int initNode);
		void cudaGetRes(vector<int> &res);
		void searchV0(int source, CudaProfiles& profile);
		void searchV1(int source, CudaProfiles& profile);
	};

	/*class CHCudaGraph : ComputeGraph {
	private:
		CudaGraph *upGraph;
		CudaGraph *downGraph;
	public:
		void search(int source, CudaProfiles& profile);
	};*/
}