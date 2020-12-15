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
		bool sort;
		bool atomic64;
		bool profile;
		int vwSize;
		int gridDim;
		int blockDim;
		int sharedLimit;
		int tileLimit;
		int distanceLimit;
		int distanceSelectLimit;
		int nodeSelectLimit;
		CudaConfigs() {}
	};
	class CudaProfiles {
	public:
		long relaxNodes;
		long relaxEdges;
		long relaxRemain;
		double kernel_time, sort_time, copy_time,select_time;
		vector<vector<int>> devF1Detail;
		vector<vector<int>> nodeDepthDetail;
		vector<int> nodeRelaxTap;
		vector<int> nodeRelaxFrec;
		int depth;
		int v, e;

		CudaProfiles() {
			relaxNodes = relaxEdges = relaxRemain = 0;
			kernel_time = sort_time = copy_time= select_time = 0.0;
			depth = 0;
		}

		void analyse() {
			if (devF1Detail.size() > 0) {
				nodeDepthDetail.resize(v);
				int level = 0;
				for (auto & x : devF1Detail) {
					for (auto & y : x) {
						nodeDepthDetail[y].push_back(level);
					}
					level++;
				}
				nodeRelaxTap.resize(v);
				nodeRelaxFrec.resize(v);
				for (int i = 0; i < v; i++) {
					vector<int> &nodeRelaxVec = nodeDepthDetail[i];
					nodeRelaxFrec[i] = nodeRelaxVec.size();
					if (nodeRelaxVec.size() > 0)
						nodeRelaxTap[i] = *nodeRelaxVec.rbegin() - *nodeRelaxVec.begin();
					else
						nodeRelaxTap[i] = 0;
				}
			}
		}
	};
	class CudaGraph {
	private:
		int v,e;
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
		void cudaGetRes(vector<int> &res);
		CudaProfiles computeAndTick(node_t source, vector<dist_t>& res, double &t);
		~CudaGraph();
	private:
		void cudaMallocMem();
		void cudaFreeMem();
		void cudaCopyMem();
		void cudaInitComputer(int initNode);
		void search(int source, CudaProfiles& profile);
		void searchV5(int source, CudaProfiles& profile);
		void searchV6(int source, CudaProfiles& profile);
		void searchV7(int source, CudaProfiles & profile);
	};
}