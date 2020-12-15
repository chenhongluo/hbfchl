#include "cudaGraph.cuh"
#include "HBFV0.cuh"
#include "HBFV1.cuh"
#include "HBFV2.cuh"
#include "HBFV3.cuh"
#include "HBFV4.cuh"
#include "HBFV5.cuh"
#include "HBFV6.cuh"
#include "HBFV7.cuh"
#include <iostream>
#include <chrono>
#include <algorithm>
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
		sort(res.begin(), res.end());
		cout << "frontier: ";
		for (int i = 0; i < size;i++) {
			cout << res[i] << " ";
		}
		cout << endl;
	}

	void CudaGraph::search(int source, CudaProfiles& profile)
	{
		// f1 relax to f2 ,devSizes 0->f1Size,1->f2Size,2->relaxEdges
		// swap(f1,f2)
		// f1Size = f2Size,f2Size = relaxEdges = 0
		int* devF1 = f1;
		int* devF2 = f2;
		long &relaxNodes = profile.relaxNodes;
		long &relaxEdges = profile.relaxEdges;
		int &depth = profile.depth;

		// init
		vector<int> hostSizes(4, 0);
		hostSizes[0] = 1;
		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		int level = 0;

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		int distanceLimit;
		int tileLimit = configs.tileLimit;

		while (1)
		{
			level++;
			depth = level;
			distanceLimit = configs.distanceLimit * level;
			// debugCudaArray<int>(devF1, hostSizes[0]);
			auto time1 = chrono::high_resolution_clock::now();
			string &kv = configs.kernelVersion;
			if (kv == "v0") {
				switchKernelV0Config(configs)
			}
			else if (kv == "v1") {
				switchKernelV1Config(configs)
			}
			else if (kv == "v2") {
				switchKernelV2Config(configs)
			}
			else if (kv == "v3") {
				switchKernelV3Config(configs)
			}
			else if (kv == "v4") {
				switchKernelV4Config(configs)
			}
			else {
				cout << "not known kernel version" << endl;
				exit(-1);
			}
			__CUDA_ERROR("GNRSearchMain Kernel");
			auto time2 = chrono::high_resolution_clock::now();
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[2];
			relaxNodes += hostSizes[0];
			//cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = 0, hostSizes[2] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			auto time3 = chrono::high_resolution_clock::now();
			profile.kernel_time += chrono::duration_cast<chrono::microseconds>(time2 - time1).count();
			profile.copy_time += chrono::duration_cast<chrono::microseconds>(time3 - time2).count();
		}
		profile.kernel_time *= 0.001;
		profile.sort_time = 0;
		profile.copy_time *= 0.001;
	}

	void CudaGraph::searchV5(int source, CudaProfiles & profile)
	{
		// f1 select to f3,remain to f2 ,devSizes 0->f1Size,1->f2Size,2->f3Size,3->relaxEdges
		// f3 relax to f2
		// swap(f1,f2)
		// f1Size = f2Size,f2Size = f3Size = relaxEdges = 0

		int* devF1 = f1;
		int* devF2 = f2;
		int* devF3 = f3;
		long &relaxNodes = profile.relaxNodes; //f3Size
		long &relaxRemain = profile.relaxRemain; //f2Size
		long &relaxEdges = profile.relaxEdges;
		int &depth = profile.depth;

		// init
		vector<int> hostSizes(4, 0);
		hostSizes[0] = 1;
		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		int level = 0;

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		int distanceLimit = 0;
		int tileLimit = configs.tileLimit;

		while (1)
		{
			level++;
			depth = level;
			distanceLimit = configs.distanceLimit * level;
			// debugCudaArray<int>(devF1, hostSizes[0]);
			auto time1 = chrono::high_resolution_clock::now();
			//<< <gdim, bdim, sharedLimit> >>
			selectNodesV5(configs)
			switchKernelV5Config(configs)

			__CUDA_ERROR("GNRSearchMain Kernel");
			auto time2 = chrono::high_resolution_clock::now();
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[3];
			relaxNodes += hostSizes[2];
			relaxRemain += hostSizes[0] - hostSizes[2];
			//cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = hostSizes[2] = hostSizes[3] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			auto time3 = chrono::high_resolution_clock::now();
			profile.kernel_time += chrono::duration_cast<chrono::microseconds>(time2 - time1).count();
			profile.copy_time += chrono::duration_cast<chrono::microseconds>(time3 - time2).count();
		}
		profile.kernel_time *= 0.001;
		profile.sort_time = 0;
		profile.copy_time *= 0.001;
	}

	void CudaGraph::searchV6(int source, CudaProfiles & profile)
	{
		// f1 select to f3,remain to f2 ,devSizes 0->f1Size,1->f2Size,2->f3Size,3->relaxEdges
		// f3 relax to f2
		// swap(f1,f2)
		// f1Size = f2Size,f2Size = f3Size = relaxEdges = 0

		int* devF1 = f1;
		int* devF2 = f2;
		int* devF3 = f3;
		long &relaxNodes = profile.relaxNodes; //f3Size
		long &relaxRemain = profile.relaxRemain; //f2Size
		long &relaxEdges = profile.relaxEdges;
		int &depth = profile.depth;

		// init
		vector<int> hostSizes(4, 0);
		hostSizes[0] = 1;
		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		int level = 0;

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		int distanceLimit = 0;
		int tileLimit = configs.tileLimit;
		int vwSize = configs.vwSize;

		while (1)
		{
			level++;
			depth = level;
			distanceLimit = configs.distanceLimit * level;
			// debugCudaArray<int>(devF1, hostSizes[0]);
			auto time1 = chrono::high_resolution_clock::now();
			//<< <gdim, bdim, sharedLimit> >>
			int vwCount = gdim * bdim / vwSize;
			tileLimit = (hostSizes[0] + vwCount - 1) / vwCount;
			if (hostSizes[0] < configs.distanceSelectLimit) {
				devF3 = devF1;
				cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
				switchKernelV6Config(configs)
				devF3 = f3;
			}
			else {
				auto time_ss = chrono::high_resolution_clock::now();
				selectNodesV6(configs)
				auto time_se = chrono::high_resolution_clock::now();
				profile.select_time += chrono::duration_cast<chrono::microseconds>(time_se - time_ss).count();
				switchKernelV6Config(configs)
			}

			__CUDA_ERROR("GNRSearchMain Kernel");
			auto time2 = chrono::high_resolution_clock::now();
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[3];
			relaxNodes += hostSizes[2];
			relaxRemain += hostSizes[0] - hostSizes[2];
			//cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = hostSizes[2] = hostSizes[3] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			auto time3 = chrono::high_resolution_clock::now();
			profile.kernel_time += chrono::duration_cast<chrono::microseconds>(time2 - time1).count();
			profile.copy_time += chrono::duration_cast<chrono::microseconds>(time3 - time2).count();
		}
		profile.kernel_time *= 0.001;
		profile.sort_time = 0;
		profile.copy_time *= 0.001;
		profile.select_time *= 0.001;
	}

	void CudaGraph::searchV7(int source, CudaProfiles & profile)
	{
		// f1 select to f3,remain to f2 ,devSizes 0->f1Size,1->f2Size,2->f3Size,3->relaxEdges
		// f3 relax to f2
		// swap(f1,f2)
		// f1Size = f2Size,f2Size = f3Size = relaxEdges = 0

		int* devF1 = f1;
		int* devF2 = f2;
		int* devF3 = f3;
		long &relaxNodes = profile.relaxNodes; //f3Size
		long &relaxRemain = profile.relaxRemain; //f2Size
		long &relaxEdges = profile.relaxEdges;
		int &depth = profile.depth;

		// init
		vector<int> hostSizes(4, 0);
		hostSizes[0] = 1;
		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		int level = 0;

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		int distanceLimit = 0;
		int tileLimit = configs.tileLimit;
		int vwSize = configs.vwSize;

		while (1)
		{
			level++;
			depth = level;
			// debugCudaArray<int>(devF1, hostSizes[0]);
			auto time1 = chrono::high_resolution_clock::now();
			//<< <gdim, bdim, sharedLimit> >>
			int vwCount = gdim * bdim / vwSize;
			tileLimit = (hostSizes[0] + vwCount - 1) / vwCount;
			if (hostSizes[0] < configs.nodeSelectLimit) {
				devF3 = devF1;
				cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
				switchKernelV6Config(configs)
				devF3 = f3;
			}
			else {
				auto time_ss = chrono::high_resolution_clock::now();
				vector<int> bb(12,0);
				bb[0] = INT_MAX;
				cudaMemcpy(devMM, &bb[0], 12 * sizeof(int), cudaMemcpyHostToDevice);
				getMinMax();
				getBulk();
				cudaMemcpy(&bb[0], devMM, 12 * sizeof(int), cudaMemcpyDeviceToHost);
				int sum = 0;
				int ii;
				for (ii = 2; ii < 12; ii++) {
					sum += bb[ii];
					if (sum > configs.nodeSelectLimit) {
						break;
					}
				}
				distanceLimit = bb[0] + ((bb[1] - bb[0]) / bulkSize + 1) * (ii - 1);

				/*printf("bb:");
				for (auto x : bb) {
					printf("%d\t", x);
				}
				printf("\n");
				printf("distanceLimit:%d\n", distanceLimit);*/
				selectNodesV6(configs)
				auto time_se = chrono::high_resolution_clock::now();
				profile.select_time += chrono::duration_cast<chrono::microseconds>(time_se - time_ss).count();
				switchKernelV6Config(configs)
			}

			__CUDA_ERROR("GNRSearchMain Kernel");
			auto time2 = chrono::high_resolution_clock::now();
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[3];
			relaxNodes += hostSizes[2];
			relaxRemain += hostSizes[0] - hostSizes[2];
			//cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = hostSizes[2] = hostSizes[3] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			auto time3 = chrono::high_resolution_clock::now();
			profile.kernel_time += chrono::duration_cast<chrono::microseconds>(time2 - time1).count();
			profile.copy_time += chrono::duration_cast<chrono::microseconds>(time3 - time2).count();
		}
		profile.kernel_time *= 0.001;
		profile.sort_time = 0;
		profile.copy_time *= 0.001;
		profile.select_time *= 0.001;
	}


	CudaGraph::~CudaGraph()
	{
		cudaFreeMem();
	}

	void CudaGraph::cudaMallocMem()
	{
		cudaMalloc((void**)&devUpOutNodes, (v + 1) * sizeof(int));
		cudaMalloc((void**)&devUpOutEdges, e * sizeof(int2));

		cudaMalloc((void**)&devSizes, 128 * sizeof(int));
		cudaMalloc((void**)&devMM, 128 * sizeof(int));
		if (configs.atomic64 == true) {
			cudaMalloc((void**)&f1, 1 * v * sizeof(int));
			cudaMalloc((void**)&f2, 1 * v * sizeof(int));
			cudaMalloc((void**)&f3, 1 * v * sizeof(int));
		}
		else {
			cudaMalloc((void**)&f1, 10 * v * sizeof(int));
			cudaMalloc((void**)&f2, 10 * v * sizeof(int));
			cudaMalloc((void**)&f3, 1 * sizeof(int));
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
		cudaFree(f3);

		cudaFree(devSizes);
		cudaFree(devMM);

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
		if (configs.kernelVersion == "v5") {
			searchV5(source, cudaProfiles);
		}
		else if (configs.kernelVersion == "v6") {
			searchV6(source, cudaProfiles);
		}
		else if (configs.kernelVersion == "v7") {
			searchV7(source, cudaProfiles);
		}
		else {
			search(source, cudaProfiles);
		}
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		cudaGetRes(res);
		cudaProfiles.analyse();
		return cudaProfiles;
	}
}