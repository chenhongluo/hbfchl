#include "cudaGraph.cuh"
#include "HBFV0.cuh"
#include "HBFV1.cuh"
#include <iostream>

namespace cuda_graph {
	cuda_graph::CudaGraph::CudaGraph(GraphWeight & _gp, CudaConfigs & _configs)
		:gp(_gp), configs(_configs), v(_gp.v), e(_gp.e)
	{
		cudaMallocMem();
		cudaCopyMem();
	}

	template<class T>
	void debugCudaArray(T* array, int size) {
		vector<T> res(size);
		cudaMemcpy(&(res[0]), array, size * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < size;i++) {
			cout << "frontier: "<< res[i] << " ";
		}
		cout << endl;
	}

	void cuda_graph::CudaGraph::search(int source)
	{
		vector<int> hostSizes(4, 0);
		hostSizes[0] = 1;
		int* devF1 = f1;
		int* devF2 = f2;
		int level = 0;
		int relaxEdges = 0;
		int relaxNodes = 0;

		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		while (1)
		{
			level++;
			debugCudaArray<int>(devF1, hostSizes[0]);
			string &kv = configs.kernelVersion;
			if (kv == "v0") {
				switchKernelV0Config(configs)
			}
			else if (kv == "v1") {
				switchKernelV1Config(configs)
			}
			else {
				cout << "not known kernel version" << endl;
				exit(-1);
			}
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[2];
			relaxNodes += hostSizes[0];
			cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = 0, hostSizes[2] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			__CUDA_ERROR("GNRSearchMain Kernel");
		}
	}

	cuda_graph::CudaGraph::~CudaGraph()
	{
		cudaFreeMem();
	}

	void cuda_graph::CudaGraph::cudaMallocMem()
	{
		cudaMalloc((void**)&devUpOutNodes, (v + 1) * sizeof(int));
		cudaMalloc((void**)&devUpOutEdges, e * sizeof(int2));

		cudaMalloc((void**)&devSizes, 4 * sizeof(int));
		if (configs.atomic64 == true) {
			cudaMalloc((void**)&f1, 1 * v * sizeof(int));
			cudaMalloc((void**)&f2, 1 * v * sizeof(int));
		}
		else {
			cudaMalloc((void**)&f1, 10 * v * sizeof(int));
			cudaMalloc((void**)&f2, 10 * v * sizeof(int));
		}

		if (configs.atomic64 == true)
			cudaMalloc((void**)&devInt2Distances, v * sizeof(int2));
		else
			cudaMalloc((void**)&devIntDistances, v * sizeof(int));
		__CUDA_ERROR("copy");
	}

	void cuda_graph::CudaGraph::cudaFreeMem()
	{
		cudaFree(f1);
		cudaFree(f2);

		cudaFree(devSizes);

		cudaFree(devUpOutNodes);
		cudaFree(devUpOutEdges);
		if (configs.atomic64 == true)
			cudaFree(devInt2Distances);
		else
			cudaFree(devIntDistances);
		cudaFree(devIntDistances);
		__CUDA_ERROR("copy");
	}

	void cuda_graph::CudaGraph::cudaCopyMem()
	{
		cudaMemcpy(devUpOutNodes, &(gp.outNodes[0]), (v + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devUpOutEdges, &(gp.outEdgeWeights[0]), e * sizeof(int2), cudaMemcpyHostToDevice);
		__CUDA_ERROR("copy");
	}

	void cuda_graph::CudaGraph::cudaInitComputer(int initNode)
	{
		//init devDistance
		int2 INF2 = make_int2(0, INT_MAX);
		int INF = INT_MAX;
		if (configs.atomic64 == true) {
			vector<int2> temp(v, INF2);
			temp[initNode] = make_int2(0, 0);
			cudaMemcpy(devInt2Distances, &temp[0], v * sizeof(int2), cudaMemcpyHostToDevice);
		}
		else {
			vector<int> temp(v, INF);
			temp[initNode] = 0;
			cudaMemcpy(devIntDistances, &temp[0], v * sizeof(int), cudaMemcpyHostToDevice);
		}
		__CUDA_ERROR("copy");
	}

	void cuda_graph::CudaGraph::cudaGetRes(vector<int>& res)
	{
		res.resize(v);
		if (configs.atomic64 == true) {
			vector<int2> res2;
			res2.resize(v);
			cudaMemcpy(&(res2[0]),devInt2Distances ,v * sizeof(int2), cudaMemcpyDeviceToHost);
			for (int i = 0; i < res.size(); i++)
				res[i] = res2[i].y;
		}
		else {
			cudaMemcpy(&(res[0]),devIntDistances , v * sizeof(int2), cudaMemcpyDeviceToHost);
		}
	}
	void CudaGraph::computeAndTick(node_t source, vector<dist_t>& res, double & t)
	{
		cudaInitComputer(source);
		search(source);
		cudaGetRes(res);
	}
}