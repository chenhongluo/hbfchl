#include "cudaGraph.cuh"
#include "HBFV0.cuh"
#include "HBFV1.cuh"
#include <iostream>
#include <chrono>

namespace cuda_graph {
	CudaGraph::CudaGraph(GraphWeight & _gp, CudaConfigs _configs)
		:gp(_gp), configs(_configs), v(_gp.v), e(_gp.e)
	{
		cudaMallocMem();
		cudaCopyMem();
	}

	template<class T>
	void debugCudaArray(T* array, int size) {
		vector<T> res(size);
		cudaMemcpy(&(res[0]), array, size * sizeof(int), cudaMemcpyDeviceToHost);
		cout << "frontier: ";
		for (int i = 0; i < size;i++) {
			cout << res[i] << " ";
		}
		cout << endl;
	}

	void CudaGraph::search(int source, CudaProfiles& profile)
	{
		vector<int> hostSizes(4, 0);
		hostSizes[0] = 1;
		int* devF1 = f1;
		int* devF2 = f2;
		int level = 0;
		long &relaxNodes = profile.relaxNodes;
		long &relaxEdges = profile.relaxEdges;
		int &depth = profile.depth;

		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		while (1)
		{
			level++;
			depth = level;
			if (configs.profile) {
				vector<int> devF1Vec(hostSizes[0]);
				cudaMemcpy(&(devF1Vec[0]), devF1, hostSizes[0] * sizeof(int), cudaMemcpyDeviceToHost);
				profile.devF1Detail.push_back(devF1Vec);
			}
			// debugCudaArray<int>(devF1, hostSizes[0]);
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
			__CUDA_ERROR("GNRSearchMain Kernel");
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[2];
			relaxNodes += hostSizes[0];
			//cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = 0, hostSizes[2] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		}
	}

	CudaGraph::~CudaGraph()
	{
		cudaFreeMem();
	}

	void CudaGraph::cudaMallocMem()
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

	void CudaGraph::cudaFreeMem()
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

	void CudaGraph::cudaCopyMem()
	{
		cudaMemcpy(devUpOutNodes, &(gp.outNodes[0]), (v + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devUpOutEdges, &(gp.outEdgeWeights[0]), e * sizeof(int2), cudaMemcpyHostToDevice);
		__CUDA_ERROR("copy");
	}

	void CudaGraph::cudaInitComputer(int initNode)
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

	void CudaGraph::cudaGetRes(vector<int>& res)
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
	CudaProfiles CudaGraph::computeAndTick(node_t source, vector<dist_t>& res, double & t)
	{
		CudaProfiles cudaProfiles;
		cudaProfiles.v = v;
		cudaProfiles.e = e;
		auto start = chrono::high_resolution_clock::now();
		cudaInitComputer(source);
		search(source, cudaProfiles);
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		cudaGetRes(res);
		cudaProfiles.analyse();
		return cudaProfiles;
	}
}