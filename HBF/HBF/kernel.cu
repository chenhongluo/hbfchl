#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/fstream.hpp>

#include <climits>
#include <cstdlib>
#include <stdio.h>
#include "fUtil.h"
#include "graph.h"
#include "cudaGraph.cuh"
#include <cub/cub.cuh>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <map>
// Declare, allocate, and initialize device-accessible pointers for input and output

using namespace graph;
using namespace cuda_graph;

#define make_sure(expression,msg)\
do{\
if(!(expression))\
{\
	__ERROR(msg) \
}\
}while(0)

vector<int> getTestNodes(int size,int seed,int v) {
	IntRandomUniform ir2 = IntRandomUniform(seed, 0, v);
	std::vector<int> vs = ir2.getValues(size);
	return vs;
}

void run(GraphWeight &graph, CudaConfigs configs,int testNodeSize);
void test(GraphWeight &graph, CudaConfigs configs,int testNodeSize);
int compareRes(vector<int>& res1, vector<int>& res2);

bool isCudaKv(string kv){
	if(kv == "V0" || kv == "V1" || kv=="V5" || kv=="V2" || kv=="V3" || kv == "V4"){
		return true;
	}else {
		return false;
	}
}

int main(int argc, char* argv[])
{
	// int  num_items = 10000;      // e.g., 7
	// int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
	// int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
	// vector<int> gggg(num_items,1);
	// cudaMalloc(&d_in, num_items * sizeof(int));
	// cudaMemcpy(d_in, &(gggg[0]), num_items * sizeof(int), cudaMemcpyHostToDevice);
	// cudaMalloc(&d_out, num_items * sizeof(int));
	// void     *d_temp_storage = NULL;
	// size_t   temp_storage_bytes = 0;
	// auto time1 = chrono::high_resolution_clock::now();
	// cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
	// cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
	// __CUDA_ERROR("GNRSearchMain Kernel");
	// auto time2 = chrono::high_resolution_clock::now();
	// cout<< chrono::duration_cast<chrono::microseconds>(time2 - time1).count() * 0.001 << endl;
	// cudaMemcpy(&(gggg[0]), d_out, num_items * sizeof(int), cudaMemcpyDeviceToHost);
	// cout<<gggg[100]<<gggg[1000]<<gggg[num_items-1]<<endl;
	// return;
	if(argc	< 2){
		justTest();
	}
	make_sure(argc>=8,"argc<4,input graph path and *.ini");
	string graphPath = argv[1];
	string action = argv[2];
	string kv = argv[3];
	int vwSize = atoi(argv[4]);
	string dsl = argv[5];
	float ds = atof(argv[6]);
	int testNodeSize = atoi(argv[7]);
	if (!boost::filesystem::exists(graphPath)) {
		std::cerr << "graph not exists." << std::endl;
		return -1;
	}

	int seed = 0;
	cout.precision(3); 
	IntRandomUniform ir = IntRandomUniform(seed, 1, 10000);
	seed = 1000;
	EdgeType userEdgeType = EdgeType::UNDEF_EDGE_TYPE;
	GraphRead* reader = getGraphReader(graphPath.c_str(), userEdgeType, ir);
	GraphWeight graphWeight(reader);
	// cout<<graphWeight.originEdges.size() << " "<<graphWeight.addEdges.size()<<endl;
	int addFlag = 1;
	if (addFlag == 1) {
		float addUsePercent = atof(argv[8]);
		vector<TriTuple> newEdges;
		int nodeAddLimit = addUsePercent * graphWeight.v;
		newEdges.reserve(graphWeight.originEdges.size() + graphWeight.addEdges.size());
		newEdges.assign(graphWeight.originEdges.begin(), graphWeight.originEdges.end());
		for (auto shortcut : graphWeight.addEdges) {
			unsigned order = graphWeight.orders[shortcut.r];
			if (order <= nodeAddLimit + 1) {
				newEdges.push_back(TriTuple(shortcut.s, shortcut.t, shortcut.w));
			}
			else if(order > graphWeight.v){
				cout << "unkown v"<< endl;
			}
		}
		int predealSize = 0;
		if(graphWeight.orders.size() > 0){
			for(int i= 0;i<graphWeight.v;i++){
				if(graphWeight.orders[i]<graphWeight.v-1){
					predealSize ++;
				}
			}
		}
		cout<<"predealSize: "<< predealSize << " per: " << (float)predealSize/graphWeight.v << endl;
		graphWeight.e = newEdges.size();
		graphWeight.originEdges = newEdges;
		graphWeight.toCSR();
		vector<TriTuple> newfilterEdges;
		newfilterEdges.reserve(graphWeight.e);
		int hhhhflag = 1;
		for(int i= 0;i<graphWeight.v;i++){
			map<int,int> fm;
			for(int j = graphWeight.outNodes[i];j<graphWeight.outNodes[i+1];j++){
				int2 index2 = graphWeight.outEdgeWeights[j];
				auto it = fm.find(index2.x);
				if(it == fm.end() || it->second > index2.y){
					fm[index2.x] = index2.y;
				}
			}
			for(auto x:fm){
				if(graphWeight.orders.size() > 0 && hhhhflag){
					if(graphWeight.orders[x.first] >= graphWeight.orders[i]){
						newfilterEdges.push_back(TriTuple(i, x.first, x.second));
					}
				}else{
					newfilterEdges.push_back(TriTuple(i, x.first, x.second));
				}
			}
		}
		graphWeight.e = newfilterEdges.size();
		graphWeight.originEdges = newfilterEdges;
	}
	else if (addFlag == 2) {

	}
	
	graphWeight.name = fileUtil::extractFileName(graphPath);
	graphWeight.toCSR();
	graphWeight.analyseSimple();
	// {
	// 	for(int kkk = 0;kkk<7;kkk++){
	// 		auto vs = graphWeight.getOutEdgesOfNode(kkk);
	// 		cout << kkk << ":"<<endl;
	// 		for(auto v: vs){
	// 			cout<<v.x << " "<<v.y<<endl;
	// 		}
	// 	}
	// }
	CudaConfigs configs = CudaConfigs(kv,vwSize,68*2,256,1024*8,dsl,ds);
	if(stringUtil::startsWith(dsl,"PBCE")){
		vector<string> vs = stringUtil::split(dsl, "_");
		configs.distanceLimitStrategy = vs[0];
		configs.PBCENUM = atoi(vs[1].c_str());
	}
	if(action == "run"){
		// if(argc == 9){
		// 	configs.dp = atoi(argv[8]);
		// }
		// configs.atomic64 = false;
		run(graphWeight,configs,testNodeSize);
	}
	else if(action == "transferGr"){
		string stemp = graphWeight.name;
		stemp +=".gr";
		graphWeight.toGr(stemp.c_str());
	}
	else if(action == "transferDDSG"){
		string stemp = graphWeight.name;
		stemp +=".ddsg";
		graphWeight.toDDSG(stemp.c_str());
	}
	else if(action == "test"){
		test(graphWeight,configs,testNodeSize);
	}
	else if(action == "nodeAllocTest"){
		if(argc == 9){
			configs.dp = atoi(argv[8]);
		}
		cout.precision(2); 
		vector<int> testNodeQueue = getTestNodes(graphWeight.v,0,graphWeight.v - 1);
		vector<int> testQueueEdges(testNodeQueue.size(),0);
		int maxmax = 1000000;
		// vector<int> testSizes;
		// for(int i=1;i<maxmax;i*=10){
		// 	for(int j=1;j<10;j++){
		// 		testSizes.push_back(i * j);
		// 	}
		// }
		int nnnn = atoi(argv[9]);
		vector<int> testSizes = {nnnn};
		for(int i=1;i<testQueueEdges.size();i++){
			testQueueEdges[i] = graphWeight.getOutDegreeOfNode(testNodeQueue[i]) + testQueueEdges[i-1];
		}
		CudaGraph* cg = new CudaGraph(graphWeight, configs);
		CudaProfiles profile;
		for(int i=0;i<testSizes.size();i++){
			int n = testSizes[i];
			if(n < testNodeQueue.size()){
				float t = 0.0;
				int k = 1;
				for(int j=0;j<k;j++){
					t += cg->nodeAllocTest(testNodeQueue, n, profile);
				}
				cout << n<<" "<<testQueueEdges[n+1] << " "<< t/k <<endl;
			}
		}
	} else if(action == "nodeWriteTest") {
		cout.precision(4); 
		vector<int> testNodeQueue = getTestNodes(testNodeSize,0,9);
		CudaGraph* cg = new CudaGraph(graphWeight, configs);
		CudaProfiles profile;
		int n = testNodeSize;
		for (int nl =1;nl <= 11;nl++){
			cg->nodeWriteTest(testNodeQueue, n,nl, profile);
		}
	} else if(action == "cacValid") {
		cout.precision(2); 
		int ssss = atoi(argv[8]);
		int cacValidPrintInterval = atoi(argv[9]);
		CudaGraph* cg = new CudaGraph(graphWeight, configs);
		CudaProfiles profile;
		cg->cacValid(ssss, cacValidPrintInterval);
	} else if(action == "none") {
		graphWeight.analyseDetail();
	}
	else {
		__ERROR("no this action")
	}
    return 0;
}


ComputeGraph* getCtGraphFromConfig(GraphWeight &graph, CudaConfigs configs) {
	if(isCudaKv(configs.kernelVersion)){
		return (ComputeGraph*)new CudaGraph(graph, configs);
	}else {
		return (ComputeGraph*)CTHostGraph::getHostGraph(graph,configs.kernelVersion);
	}
}

int compareRes(vector<int>& res1, vector<int>& res2)
{
	if (res1.size() != res2.size()) {
		return -2;
	}
	for (int i = 0; i < res1.size(); i++) {
		if (res1[i] != res2[i]) {
			cout<< i <<" not correct" << endl;
			return -1;
		}
	}
	return 0;
}

void test(GraphWeight &graph,CudaConfigs configs,int testNodeSize)
{
	ComputeGraph* cmpCt = (ComputeGraph*)CTHostGraph::getHostGraph(graph, "Dijkstra");
	vector<node_t> testNodes = getTestNodes(testNodeSize, 0, graph.v);
	vector<dist_t> dis1(graph.v), dis2(graph.v);
	double t1, t2;
	ComputeGraph* ct = getCtGraphFromConfig(graph, configs);

	for (int i = 0; i < testNodes.size(); i++) {
		ct->computeAndTick(testNodes[i], dis1, t1);
		cmpCt->computeAndTick(testNodes[i], dis2, t2);
		for(int j = 0;j<dis1.size();j++){
			// cout<<dis1[j]<<endl;
		}
		if (compareRes(dis1, dis2) == 0) {
			cout << "the "<< i << " source node: " << testNodes[i] << " is correct" << endl;
		}
		else {
			cout << "the " << i << " source node: " << testNodes[i] << " is wrong" << endl;
		}
	}
}
void run(GraphWeight &graph, CudaConfigs configs,int testNodeSize)
{
	bool printDeatil = true;

	vector<node_t> testNodes = getTestNodes(testNodeSize, 0, graph.v);
	for (int i = 0;i<testNodeSize;i++){
		testNodes[i] = 0;
	}
	vector<dist_t> dis(graph.v);
	double t, allt = 0.0, allkt = 0.0, allst = 0.0, allct = 0.0 ,allslt = 0.0;
	double allRN = 0.0, allRE = 0.0, allDP = 0.0 ,allRM = 0.0;
	double allDl = 0.0,dl;
	cout.precision(2);
	if(!isCudaKv(configs.kernelVersion)){
		ComputeGraph* ct = getCtGraphFromConfig(graph, configs);
		for (int i = 0; i < testNodes.size(); i++) {
			ct->computeAndTick(testNodes[i], dis, t);
			// cout << "Relax Source: " << testNodes[i] << "\trelaxNodes: " << 0 << "\trelaxEdges: " << 0 << "\tuseTime: " << t << endl;
			allt += t;
		}
	}
	else{
		ComputeGraph* cg = getCtGraphFromConfig(graph, configs);
		for (int i = 0; i < testNodes.size(); i++) {

			CudaProfiles pf = *(CudaProfiles*)cg->computeAndTick(testNodes[i], dis, t);
			dl = 0.0;
			// cout<< "res:" << dis[100] << dis[1000]  << dis[10000] << endl;
			for(auto &x:dis){
				if(x!=INT_MAX && x>dl){
					dl = x;
				}
			}
			dl = dl / pf.depth;
			if (printDeatil) {
				// cout << "Relax Source: " << testNodes[i]
				// 	<< "\trelaxNodes: " << pf.relaxNodes << "\trelaxNodesDivV: " << (double)pf.relaxNodes / graph.v
				// 	<< "\trelaxEdges: " << pf.relaxEdges << "\trelaxEdgesDivE: " << (double)pf.relaxEdges / graph.e
				// 	<< "\tdepth: " << pf.depth
				// 	<< "\tuseTime: " << t
				// 	<< "\tkernelTime: " << pf.kernel_time << "\tcacTime: " << pf.cac_time << "\tcopyTime: " << pf.copy_time 
				// 	<< "\tselectTime: " << pf.select_time
				// 	<< "\trelaxRemain: " << pf.relaxRemain
				// 	<<"\tdl: "<<dl
				// 	<< endl;
				
			}
			allDl += dl;
			allt += t;
			allRN += pf.relaxNodes;
			allRE += pf.relaxEdges;
			allDP += pf.depth;
			allkt += pf.kernel_time;
			allst += pf.cac_time;
			allct += pf.copy_time;
			allslt += pf.select_time;
			allRM += pf.relaxRemain;
		}
	}

	allDl /= testNodes.size();
	allRN /= testNodes.size();
	allRE /= testNodes.size();
	allt /= testNodes.size();
	allDP /= testNodes.size();
	allkt /= testNodes.size();
	allst /= testNodes.size();
	allct /= testNodes.size();
	allRM /= testNodes.size();
	allslt /= testNodes.size();

	cout << " Avg Profile: " << endl;
	// cout << "\trelaxNodes: " << allRN << "\trelaxNodesDivV: " << allRN / graph.v
	// 	<< "\trelaxNodesPerKernel: " << allRN/allDP
	// 	<< "\trelaxEdges: " << allRE << "\trelaxEdgesDivE: " << allRE / graph.e
	// 	<< "\trelaxDepth: " << allDP
	// 	<< "\trelaxEdgesPerKernel: " << allRE/allDP
	// 	<< "\tuseTime: " << allt << " " << allkt + allct << " " << allkt
	// 	<< "\tkernelTime: " << allkt << "\tcacTime: " << allst << "\tcopyTime: " << allct
	// 	<< "\tselectTime: " << allslt
	// 	<< "\trelaxRemain: " << allRM
	// 	<< "\tDL: "<< allDl
	// 	<< endl;
	cout << "relaxNodesDivV: " << allRN / graph.v
		<< "\trelaxDepth: " << allDP
		<< "\tkernelTime: " << allkt 
		<< "\tcacTime: " << allst 
		<< "\tcopyTime: " << allct
		<< "\tselectTime: " << allslt
		<< "\tDL: "<< allDl
		<< endl;
	cout.precision(2);
	cout << "here::: " << allkt << " " << allst << " " << allslt << " " << allRN / graph.v << " " << allDP << " " << configs.distanceLimit << endl;
}