#include "WorkList.cuh"
#include "cudaGraph.cuh"
#include "HBFV0.cuh"
#include "HBFV1.cuh"
#include "HBFV2.cuh"
#include "HBFV3.cuh"
#include "HBFV5.cuh"
#include "HBFV6.cuh"
#include "HBFV7.cuh"
#include "HBFV8.cuh"
#include "HBFV9.cuh"
#include "mutex.cuh"
#include "memManager.cuh"
#include "HBFV11.cuh"
#include "HBFV12.cuh"
#include "HBFV13.cuh"
#include "WK.cuh"
#include "ana.cuh"
#include "fUtil.h"
#include "gbar.cuh"
#include <climits>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <mutex>

#define cacValidInternal() do{ \
	initValidRes(); \
	switchKernelCacValid() \
	getValidRes(level,printInterval,distanceLimit); \
}while(0)

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

	float getStrictDis(float dl,int bcea,int bcek,int level){
		int a = level/bcea;
		int b = level%bcea;
		return pow(b,bcek) * dl / pow(bcea,bcek-1) + a * bcea *dl;
	}

	void CudaGraph::searchV0(int source, CudaProfiles& profile)
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
		int dp = configs.dp;

		while (1)
		{
			level++;
			depth = level;
			// debugCudaArray<int>(devF1, hostSizes[0]);
			auto time1 = chrono::high_resolution_clock::now();
			string &kv = configs.kernelVersion;
			if (configs.distanceLimitStrategy == "none") {
				cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			switchKernelV0Config(configs)
			__CUDA_ERROR("GNRSearchMain Kernel");
			auto time2 = chrono::high_resolution_clock::now();
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[3];
			relaxNodes += hostSizes[0];
			//cout << "level: " << level << "\tf1Size: " << hostSizes[0] << "\trelaxEdges: " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = 0, hostSizes[2] = 0, hostSizes[3] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			auto time3 = chrono::high_resolution_clock::now();
			profile.kernel_time += chrono::duration_cast<chrono::microseconds>(time2 - time1).count();
			profile.copy_time += chrono::duration_cast<chrono::microseconds>(time3 - time2).count();
		}
		profile.kernel_time *= 0.001;
		profile.cac_time = 0;
		profile.copy_time *= 0.001;
	}

	void CudaGraph::searchV1(int source, CudaProfiles & profile)
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
		float distanceLimit = 0.0;
		float delta = configs.distanceLimit;
		int vwSize = configs.vwSize;
		int dp = configs.dp;

		while (1)
		{
			level++;
			depth = level;
			if(configs.distanceLimitStrategy == "normal"){
				distanceLimit += configs.distanceLimit;
			}else if(configs.distanceLimitStrategy == "delta"){
				distanceLimit = delta;
			}else if(configs.distanceLimitStrategy == "strict"){
				distanceLimit = getStrictDis(configs.distanceLimit,configs.bcea,configs.bcek,level);
			}
			auto time1 = chrono::high_resolution_clock::now();
			// if(hostSizes[2] < 68 * 32){
			// 	vwSize = 32;
			// }else if(hostSizes[2] < 68 * 32 * 2){
			// 	vwSize = 16;
			// }else if(hostSizes[2] < 68 * 32 * 4){
			// 	vwSize = 8;
			// }
			if (configs.distanceLimitStrategy == "none") {
				devF3 = devF1;
				cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
				switchKernelV2Config(configs)
				devF3 = f3;
			}
			else if (configs.distanceLimitStrategy == "normal" || 
			configs.distanceLimitStrategy == "strict"
		){
				auto time_ss = chrono::high_resolution_clock::now();
				selectNodesV1(configs)
				__CUDA_ERROR("GNRSearchMain Kernel");
				auto time_se = chrono::high_resolution_clock::now();
				profile.select_time += chrono::duration_cast<chrono::microseconds>(time_se - time_ss).count();
				switchKernelV2Config(configs)
			}else if(configs.distanceLimitStrategy == "perfect"){
				auto time_ss = chrono::high_resolution_clock::now();
				selectNodestPerfectV1(configs)
				__CUDA_ERROR("GNRSearchMain Kernel");
				auto time_se = chrono::high_resolution_clock::now();
				profile.select_time += chrono::duration_cast<chrono::microseconds>(time_se - time_ss).count();
				switchKernelV2Config(configs)
			} if(configs.distanceLimitStrategy == "delta"){
				auto time_ss = chrono::high_resolution_clock::now();
				selectNodesDeltaV1(configs)
				__CUDA_ERROR("GNRSearchMain Kernel");
				auto time_se = chrono::high_resolution_clock::now();
				profile.select_time += chrono::duration_cast<chrono::microseconds>(time_se - time_ss).count();
				switchKernelV2Config(configs)
			}

			__CUDA_ERROR("GNRSearchMain Kernel");
			auto time2 = chrono::high_resolution_clock::now();
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			relaxEdges += hostSizes[3];
			relaxNodes += hostSizes[2];
			relaxRemain += hostSizes[0] - hostSizes[2];
			if(hostSizes[2] == 0){
				delta += configs.distanceLimit;
			}
			// if(hostSizes[2] > 15000){
			// 	distanceLimit -= configs.distanceLimit;
			// }
			// if(hostSizes[2] < 5000 && depth > 20){
			// 	distanceLimit += 100 *configs.distanceLimit;
			// }
			// cout << "level: " << level << " " << hostSizes[0] << " " << hostSizes[1] << " " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = hostSizes[2] = hostSizes[3] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
			auto time3 = chrono::high_resolution_clock::now();
			// cout<<"kernel time:" <<hostSizes[0] <<" "<< chrono::duration_cast<chrono::microseconds>(time2 - time1).count() << endl;
			profile.kernel_time += chrono::duration_cast<chrono::microseconds>(time2 - time1).count();
			profile.copy_time += chrono::duration_cast<chrono::microseconds>(time3 - time2).count();
		}
		profile.kernel_time *= 0.001;
		profile.cac_time = 0;
		profile.copy_time *= 0.001;
		profile.select_time *= 0.001;
	}

	int get_GPU_Rate()
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,0);
		return deviceProp.clockRate;
	}

	void CudaGraph::searchV2(int source, CudaProfiles & profile)
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
		vector<int> hostSizes(10, 0);
		vector<float> timess(10,0);
		hostSizes[0] = 1;
		cudaMemcpy(devSizes, &(hostSizes[0]), 10 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		int level = 0;

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		float distanceLimit = configs.distanceLimit;
		float delta = configs.distanceLimit;
		int vwSize = configs.vwSize;
		int dp = configs.dp;
		float initDL = distanceLimit; 
		static GlobalBarrierLifetime gb;
		int clockFrec = get_GPU_Rate();
		gb.Setup(gdim);
		int strategyNum = 0;
		if (configs.distanceLimitStrategy == "delta"){
			strategyNum = -1;
		} else if(configs.distanceLimitStrategy == "PBCE"){
			// strategyNum = (gdim*bdim) / (pow(2,int(log2(e/v)) + 1));
			strategyNum = configs.PBCENUM;
		} else if(configs.distanceLimitStrategy == "perfect"){
			strategyNum = -2;
		} else if(configs.distanceLimitStrategy == "none"){
			strategyNum = -3;
		}
		// vwSize = pow(2,int(log2(E/V)) + 1);

		auto time1 = chrono::high_resolution_clock::now();
		// cout<<gdim << " "<<bdim <<endl;
		float elapsed_time;
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
		cudaEventRecord(start_event, 0);
		// kernelV10Atmoic64(gdim, bdim,sharedLimit,initDL,distanceLimit,gb);
		kernelV8Atmoic64(gdim, bdim,sharedLimit,initDL,distanceLimit,gb);
		// kernelV9Atmoic64(gdim, bdim,sharedLimit,initDL,distanceLimit,gb);
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
	
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		profile.kernel_time = elapsed_time;
		// cout<<"kernelTime:"<<profile.kernel_time<<endl;

		cudaMemcpy(&(hostSizes[0]), devSizes, 10 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&(timess[0]), times, 10 * sizeof(float), cudaMemcpyDeviceToHost);
		profile.depth = hostSizes[7];
		profile.relaxNodes = hostSizes[6];
		profile.select_time = timess[0];
		profile.cac_time = timess[1];
		profile.copy_time = profile.kernel_time - profile.cac_time - profile.select_time;
	}

	void CudaGraph::searchV3(int source, CudaProfiles & profile)
	{
		// f1 select to f3,remain to f2 ,devSizes 0->f1Size,1->f2Size,2->f3Size,3->relaxEdges
		// f3 relax to f2
		// swap(f1,f2)
		// f1Size = f2Size,f2Size = f3Size = relaxEdges = 0

		long &relaxNodes = profile.relaxNodes; //f3Size
		long &relaxRemain = profile.relaxRemain; //f2Size
		long &relaxEdges = profile.relaxEdges;
		int &depth = profile.depth;

		// init
		int qnum = 32;
		WorkListsPlain workLists(qnum + 2,v);
		int ggsize = 1;
		cudaMemcpy(workLists.data, &source, 1 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(workLists.qsizes, &ggsize, 1 * sizeof(int), cudaMemcpyHostToDevice);

		vector<int> hostSizes(128, 0);
		vector<float> timess(10,0);
		hostSizes[0] = 1;
		cudaMemcpy(devSizes, &(hostSizes[0]), 128 * sizeof(int), cudaMemcpyHostToDevice);
		cudaStream_t stream1,stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);


		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		float distanceLimit = configs.distanceLimit;
		float delta = configs.distanceLimit;
		int vwSize = configs.vwSize;
		int dp = configs.dp;
		float initDL = distanceLimit; 
		static GlobalBarrierLifetime gb;
		int clockFrec = get_GPU_Rate();
		gb.Setup(gdim);
		int strategyNum = 0;
		if (configs.distanceLimitStrategy == "delta"){
			strategyNum = -1;
		} else if(configs.distanceLimitStrategy == "PBCE"){
			// strategyNum = (gdim*bdim) / (pow(2,int(log2(e/v)) + 1));
			strategyNum = configs.PBCENUM;
		} else if(configs.distanceLimitStrategy == "perfect"){
			strategyNum = -2;
		}
		int kernelFlag = 0;
		if(configs.kernelVersion == "V4"){
			kernelFlag = 1;
		}
		if(configs.kernelVersion == "V5"){
			kernelFlag = 2;
		}
		
		auto time1 = chrono::high_resolution_clock::now();
		// cout<<gdim << " "<<bdim <<endl;
		float elapsed_time;
		cudaEvent_t start_event, stop_event;
		cudaEventCreate(&start_event);
		cudaEventCreate(&stop_event);
		cudaEventRecord(start_event, 0);
		#if Profile
			unsigned long long maxTimeCounts = clockFrec * 500; 
			printDetail<<<1,1,0,stream1>>>(devSizes,clockFrec,maxTimeCounts);
		#endif

		if(strategyNum == 0){
			kernelV11Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb);
		}else if(strategyNum > 0){
			if(configs.atomic64 == true){
				kernelV12Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb);
			}else{
				kernelV13Atmoic64(gdim, bdim,sharedLimit,workLists,distanceLimit,gb);
			}
		}
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
		profile.kernel_time = elapsed_time;
		__CUDA_ERROR("run");

		cudaMemcpy(&(hostSizes[0]), devSizes, 128 * sizeof(int), cudaMemcpyDeviceToHost);
		profile.depth = hostSizes[7];
		profile.relaxNodes = hostSizes[6];
		cudaMemcpy(&(timess[0]), times, 10 * sizeof(float), cudaMemcpyDeviceToHost);
		profile.select_time = timess[0];
		profile.cac_time = timess[1];
		workLists.deleteData();
		__CUDA_ERROR("run");
		
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
	}

	bool myfunction (ValidStruct i,ValidStruct j) { return (i.d<j.d); }

	void CudaGraph::initValidRes(){
		vector<int> zs(5,0);
		cudaMemcpy(validSizes, &zs[0], 5 * sizeof(int), cudaMemcpyHostToDevice);
	}
	void CudaGraph::getValidRes(int level,int printInterval,float distanceLimit){
		vector<int> zs(5,0);
		vector<int> hostSizes(4, 0);
		cudaMemcpy(&(zs[0]), validSizes, 5 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
		vector<ValidStruct> vRes(zs[0]);
		cudaMemcpy(&(vRes[0]), validRes, zs[0] * sizeof(ValidStruct), cudaMemcpyDeviceToHost);
		int vs = 0,nvs = 0;
		for(auto &x: vRes){
			if(x.flag == 1){
				vs ++;
			}else if(x.flag == 0){
				nvs ++;
			}
		}
		cout << "validInfo: " << level << " " << hostSizes[2] << " " << vs << " " << nvs <<" " << zs[0] <<
		" " << zs[1] << " " << zs[2] << " " << zs[3] << " " << zs[4] <<" "<< distanceLimit <<" "<< hostSizes[0] << endl;
		if(level % printInterval == 0){
			std::sort (vRes.begin(), vRes.end(), myfunction);
			for(auto &x: vRes){
				cout << x.d << " " << x.flag << endl;
			}
		}
	}

	int CudaGraph::getTrueDistance(int source){
		CudaProfiles cudaProfiles;
		cudaInitComputer(source);
		// string temp = configs.distanceLimitStrategy;
		// configs.distanceLimitStrategy == "PBCE";
		searchV0(source, cudaProfiles);
		// configs.distanceLimitStrategy = temp;
		cudaMemcpy(devTrueInt2Distances, devInt2Distances, v * sizeof(int2), cudaMemcpyDeviceToDevice);
		return cudaProfiles.depth;
	}

	void CudaGraph::cacValid(node_t source,int printInterval){
		int realDepth = getTrueDistance(source);
		if(printInterval <= 0){
			printInterval = INT_MAX;
		}else{
			printInterval = (realDepth-1) / printInterval;
		}
		if(printInterval == 0){
			printInterval = 1;
		}
		cout <<"realDepth: " << realDepth << " " << printInterval << endl;
		cudaInitComputer(source);
		int* devF1 = f1;
		int* devF2 = f2;
		int* devF3 = f3;

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
		float delta = configs.distanceLimit;
		float distanceLimit = 0;
		int vwSize = configs.vwSize;
		int dp = configs.dp;

		while (1)
		{
			level++;
			if(configs.distanceLimitStrategy == "normal" || configs.distanceLimitStrategy == "PBCE"){
				distanceLimit += configs.distanceLimit;
			}else if(configs.distanceLimitStrategy == "delta"){
				distanceLimit = delta;
			}else if(configs.distanceLimitStrategy == "strict"){
				distanceLimit = getStrictDis(configs.distanceLimit,configs.bcea,configs.bcek,level);
			}
			if (configs.distanceLimitStrategy == "none") {
				devF3 = devF1;
				cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
				cacValidInternal();
				switchKernelV2Config(configs)
				devF3 = f3;
			}
			else if (configs.distanceLimitStrategy == "normal" || 
			configs.distanceLimitStrategy == "strict"){
				selectNodesV1(configs)
				cacValidInternal();
				switchKernelV2Config(configs)
			}else if(configs.distanceLimitStrategy == "perfect"){
				selectNodestPerfectV1(configs)
				cacValidInternal();
				__CUDA_ERROR("GNRSearchMain Kernel");
				switchKernelV2Config(configs)
			} if(configs.distanceLimitStrategy == "delta"){
				distanceLimit = delta;
				selectNodesDeltaV1(configs)
				cacValidInternal();
				__CUDA_ERROR("GNRSearchMain Kernel");
				switchKernelV2Config(configs)
			} else if(configs.distanceLimitStrategy == "PBCE"){
				selectNodesV1(configs)
				cacValidInternal();
				switchKernelV2Config(configs)
			} 
			__CUDA_ERROR("GNRSearchMain Kernel");
			std::swap(devF1, devF2);
			cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
			if(hostSizes[2] == 0){
				delta += configs.distanceLimit;
				distanceLimit = delta;
			}
			if(configs.distanceLimitStrategy == "PBCE" && hostSizes[2] > configs.PBCENUM){
				distanceLimit -= configs.distanceLimit;
			}
			// cout << "level: " << level << " " << hostSizes[0] << " " << hostSizes[1] << " " << hostSizes[2] << endl;
			hostSizes[0] = hostSizes[1], hostSizes[1] = hostSizes[2] = hostSizes[3] = 0;
			if (hostSizes[0] == 0) break;
			cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		}
	}

	float CudaGraph::nodeAllocTest(vector<int> sources,int n, CudaProfiles & profile)
	{
		// init
		int* devF1 = f1;
		int* devF2 = f2;
		int* devF3 = f3;
		vector<int> hostSizes(4, 0);
		hostSizes[0] = n;
		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &(sources[0]), n * sizeof(int), cudaMemcpyHostToDevice);
		int2 INF2 = make_int2(0, INT_MAX);
		vector<int2> temp(v, INF2);
		for(int i=0;i<n;i++){
			temp[sources[i]]=make_int2(0, 0);
		}
		cudaMemcpy(devInt2Distances, &temp[0], v * sizeof(int2), cudaMemcpyHostToDevice);
		int level = 0;
		devF3 = devF1;
		cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
		vector<int2> BF;
		int BFSize = 0;
		for(int i=0;i<n;i++){
			BFSize+=gp.getOutDegreeOfNode(sources[i]);
		}
		BF.resize(BFSize);
		int BFgg = 0;
		for(int i=0;i<n;i++){
			for(int j=0;j<gp.getOutDegreeOfNode(sources[i]);j++){
				BF[BFgg++] = make_int2(sources[i],j);
			}
		}
		cout << BFSize << " "<<BFgg<<endl;
		cudaMemcpy(devBF, &BF[0], BFSize * sizeof(int2), cudaMemcpyHostToDevice);
		cudaMemcpy(devBFSize, &BFSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
		vector<int> pf1,pf2,pf3;
		for(int i=0;i<n;i++){
			if(gp.getOutDegreeOfNode(sources[i])<32){
				pf1.push_back(sources[i]);
			} else if(gp.getOutDegreeOfNode(sources[i])<256){
				pf2.push_back(sources[i]);
			} else{
				pf3.push_back(sources[i]);
			}
		}
		int pfSize = pf1.size();
		cudaMemcpy(devPF1, &pf1[0], pfSize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devPFSize1, &pfSize, 1 * sizeof(int), cudaMemcpyHostToDevice);


		pfSize = pf2.size();
		cudaMemcpy(devPF2, &pf2[0], pfSize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devPFSize2, &pfSize, 1 * sizeof(int), cudaMemcpyHostToDevice);


		pfSize = pf3.size();
		cudaMemcpy(devPF3, &pf3[0], pfSize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devPFSize3, &pfSize, 1 * sizeof(int), cudaMemcpyHostToDevice);
		__CUDA_ERROR("GNRSearchMain Kernel");


		cout << pf1.size() << " "<<pf2.size()<<" "<<pf3.size()<<endl;

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		int distanceLimit = 0;
		int vwSize = configs.vwSize;
		int dp = configs.dp;
		cudaStream_t stream1,stream2,stream3;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		cudaStreamCreate(&stream3);

		auto time1 = chrono::high_resolution_clock::now();
		if (configs.kernelVersion == "V0") {
			switchKernelV0Config(configs) //HBF
		}
		else if (configs.kernelVersion == "V1") {
			switchKernelV1Config(configs)
		}else if (configs.kernelVersion == "V2") {
			switchKernelV2Config(configs) //VW
		}
		else if (configs.kernelVersion == "V3") {
			switchKernelV3Config(configs) // DW
		}
		else if (configs.kernelVersion == "V5") {
			switchKernelV5Config(configs) // VW + DP
		}else if(configs.kernelVersion == "V6") {
			switchKernelV6Config(configs) //perfect
		}
		else if(configs.kernelVersion == "V7") {
			// cudaMemcpy(devPFSize1, devSizes+2, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
			// kernelV7Atmoic64(32,68*4, 256,1024*8,stream1, devF3, devPFSize1)
			kernelV7Atmoic64(32,68*6, 256,1024*8,stream1, devPF1, devPFSize1)
			kernelV7Atmoic64(256,68*2, 256,1024*8,stream2, devPF2, devPFSize2)
			kernelV7Atmoic64(1024,35, 1024,1024*16,stream3, devPF3, devPFSize3)
		}
		else{
			__ERROR("no this cuda kernelversion")
		}
		__CUDA_ERROR("GNRSearchMain Kernel");
		auto time2 = chrono::high_resolution_clock::now();
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
		cudaStreamDestroy(stream3);
		float t =  chrono::duration_cast<chrono::microseconds>(time2 - time1).count() * 0.001;
		cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
		cout << hostSizes[0] << " " << hostSizes[1] << " "<<hostSizes[2] << " "<<hostSizes[3] << " " << t <<endl;
		return t;
	}

	__global__ void justTestSomething(gpu_mutex mutex,int *dev_v){
		mutex.lock();
		*dev_v = *(int volatile*)dev_v+1;
		mutex.unlock();
	}
	__global__ void justTestSomething2(memManager manager){
		int v  = manager.getBlock();
		int *p = manager.getDevicePoint(v);
		p[0] = 1;
		p[1] = 1;
		p[2] = 4;
		printf("%d,%d\n",blockIdx.x*blockDim.x + threadIdx.x,v);
		manager.release(v);
	}

	void justTest(){
		// gpu_mutex mutex;
		// int* dev_v = 0;
		// int v = 0;
		// cudaMalloc((void**)&dev_v, 1 * sizeof(int));
		// cudaMemcpy(dev_v, &v, 1 * sizeof(int), cudaMemcpyHostToDevice);
		// justTestSomething<<<16,1024>>>(mutex,dev_v);
		// cudaMemcpy(&v, dev_v, 1 * sizeof(int), cudaMemcpyDeviceToHost);
		// cout <<v <<endl;
		memManager manager(4*4096,4);
		justTestSomething2<<<1,32>>>(manager);
		manager.debug();
	}

	float CudaGraph::nodeWriteTest(vector<int> sources,int n,int nl, CudaProfiles & profile)
	{
		// init
		int* devF1 = f1;
		int* devF2 = f2;
		int* devF3 = f3;
		vector<int> hostSizes(4, 0);
		int gggg = 0;
		for(int kkk =0;kkk<n;kkk++){
			if(sources[kkk] < nl){
				gggg++;
			}
		}
		hostSizes[0] = n;
		cudaMemcpy(devSizes, &(hostSizes[0]), 4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devF1, &(sources[0]), n * sizeof(int), cudaMemcpyHostToDevice);
		int level = 0;
		devF3 = devF1;
		cudaMemcpy(devSizes + 2, devSizes, 1 * sizeof(int), cudaMemcpyDeviceToDevice);

		// config
		int gdim = configs.gridDim;
		int bdim = configs.blockDim;
		int sharedLimit = configs.sharedLimit;
		int distanceLimit = 0;
		int vwSize = configs.vwSize;

		auto time1 = chrono::high_resolution_clock::now();
		if (configs.kernelVersion == "V0") {
			switchWriteKernel(WriteKernel1)
		}
		else if (configs.kernelVersion == "V1") {
			switchWriteKernel(WriteKernel2)
		}else if (configs.kernelVersion == "V2") {
			switchWriteKernel(WriteKernel3)
		}
		else if (configs.kernelVersion == "V3") {
			switchWriteKernel(WriteKernel4)
		}
		else{
			__ERROR("no this cuda kernelversion")
		}
		__CUDA_ERROR("GNRSearchMain Kernel");
		auto time2 = chrono::high_resolution_clock::now();
		float t =  chrono::duration_cast<chrono::microseconds>(time2 - time1).count() * 0.001;
		cudaMemcpy(&(hostSizes[0]), devSizes, 4 * sizeof(int), cudaMemcpyDeviceToHost);
		cout << n << " "<<nl << " "<< hostSizes[1] << " "<<hostSizes[2] << " "<< gggg << " "<<hostSizes[3] << " " << t <<endl;
		return t;
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
		cudaMalloc((void**)&times, 10 * sizeof(float));
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

		cudaMalloc((void**)&devIntDistances, v * sizeof(int));
		cudaMalloc((void**)&devInt2Distances, v * sizeof(int2));


		
		cudaMalloc((void**)&devTrueInt2Distances, v * sizeof(int2));
		cudaMalloc((void**)&validRes, v * sizeof(int2));
		cudaMalloc((void**)&validSizes, 128 * sizeof(int));	
		cudaMalloc((void**)&devBF,e*sizeof(int2));
		cudaMalloc((void**)&devBFSize,1*sizeof(int));

		cudaMalloc((void**)&devPF1,v*sizeof(int));
		cudaMalloc((void**)&devPFSize1,1*sizeof(int));
		cudaMalloc((void**)&devPF2,v*sizeof(int));
		cudaMalloc((void**)&devPFSize2,1*sizeof(int));
		cudaMalloc((void**)&devPF3,v*sizeof(int));
		cudaMalloc((void**)&devPFSize3,1*sizeof(int));
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
		cudaFree(devInt2Distances);
		cudaFree(devIntDistances);

		cudaFree(devTrueInt2Distances);
		cudaFree(validRes);
		cudaFree(validSizes);
		cudaFree(devBF);
		cudaFree(devBFSize);
		cudaFree(devPF1);
		cudaFree(devPFSize2);
		cudaFree(devPF2);
		cudaFree(devPFSize2);
		cudaFree(devPF3);
		cudaFree(devPFSize3);
		cudaFree(times);
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
		__CUDA_ERROR("copy0");

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
			cudaMemcpy(&(res[0]),devIntDistances , v * sizeof(int), cudaMemcpyDeviceToHost);
		}
	}
	void* CudaGraph::computeAndTick(node_t source, vector<dist_t>& res, double & t)
	{
		if(configs.distanceLimitStrategy == "perfect"){
			getTrueDistance(source);
		}
	#if Profile
		getTrueDistance(source);
	#endif
		CudaProfiles cudaProfiles;
		cudaProfiles.v = v;
		cudaProfiles.e = e;
		auto start = chrono::high_resolution_clock::now();
		// int size = 100;
		// vector<int2> reses(size);
		// cudaMemcpy(&(reses[0]), devTrueInt2Distances, size * sizeof(int2), cudaMemcpyDeviceToHost);
		// for (int i = 0; i < size;i++) {
		// 	cout << reses[i].y << " ";
		// }
		// cout << endl;
		__CUDA_ERROR("at1");
		cudaInitComputer(source);
		__CUDA_ERROR("at2");
		if (configs.kernelVersion == "V0") {
			searchV0(source, cudaProfiles);
		}
		else if (configs.kernelVersion == "V1") {
			searchV1(source, cudaProfiles);
		}
		else if (configs.kernelVersion == "V2") {
			searchV2(source, cudaProfiles);
		}
		else if (configs.kernelVersion == "V3" || configs.kernelVersion == "V4" || configs.kernelVersion == "V5") {
			searchV3(source, cudaProfiles);
		}
		else{
			__ERROR("no this cuda kernelversion")
		}
		__CUDA_ERROR("at3");
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		cudaGetRes(res);
		__CUDA_ERROR("at4");
		cudaProfiles.cac();
		return new CudaProfiles(cudaProfiles);
	}
}