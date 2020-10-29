#pragma once
#include "graph.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::cerr << std::endl << " CUDA error   "
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
		string kernel;
		CudaConfigs() {}
	};
	class CudaGraph {
	private:
		int v,e;
		GraphWeight &gp;
	public:
		int* f1, *f2;
		int *devSizes;

		int *devUpOutNodes;
		int2 *devUpOutEdges;

		int *devIntDistances;
		int2 *devInt2Distances;

		CudaConfigs configs;
		CudaGraph(GraphWeight & _gp, CudaConfigs & _configs);
		void search(int source);
		void cudaGetRes(vector<int> &res);
		~CudaGraph();
	private:
		void cudaMallocMem();
		void cudaFreeMem();
		void cudaCopyMem();
		void cudaInitComputer(int initNode);
	};
}