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

	void cuda_graph::CudaGraph::search(int source)
	{
		int f1Size = 1, f2Size, f3Size, f4Size;
		f2Size = f3Size = f4Size = 0;
		vector<int> hostSizes(4, 0);
		int* devF1 = f1;
		int* devF2 = f2;
		int level = 0;
		int relaxEdges = 0;
		int relaxNodes = 0;

		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		while (1)
		{
			level++;
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
			f1Size = hostSizes[0], f2Size = hostSizes[1], f3Size = hostSizes[2];
			if (f1Size == 0) break;
			relaxEdges += f3Size;
			relaxNodes += f1Size;
			hostSizes[0] = f2s, hostSizes[1] = 0, hostSizes[2] = 0;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			__CUDA_ERROR("GNRSearchMain Kernel");
			cout << "level: " << level << "\tf1Size: " << f1s << "\trelaxEdges: " << re << endl;
		}
	}

	cuda_graph::CudaGraph::~CudaGraph()
	{
		cudaFreeMem();
	}

	void cuda_graph::CudaGraph::cudaMallocMem()
	{
		cudaMalloc((void**)&f1, 1 * v * sizeof(int));
		cudaMalloc((void**)&f2, 1 * v * sizeof(int));

		cudaMalloc((void**)&devSizes, 4 * sizeof(int));
		if (configs.atomic64 == true) 
			cudaMalloc((void**)&devUpOutNodes, (v + 1) * sizeof(int));
		else
			cudaMalloc((void**)&devUpOutNodes, 10 * (v + 1) * sizeof(int));
		cudaMalloc((void**)&devUpOutEdges, e * sizeof(int2));

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

		vector<int> nodes;
		nodes.push_back(initNode);
		cudaMemcpy(f1, &nodes[0], 1 * sizeof(int), cudaMemcpyHostToDevice);
		vector<int> sizes(4, 0);
		sizes[0] = 1;
		cudaMemcpy(devSizes, &sizes[0], 4 * sizeof(int), cudaMemcpyHostToDevice);
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
}