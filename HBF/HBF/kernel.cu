#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/fstream.hpp>

#include <climits>
#include <cstdlib>
#include <stdio.h>
#include "graph.h"
#include "cudaGraph.cuh"
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
	if(kv == "V0" || kv == "V1"){
		return true;
	}else {
		return false;
	}
}

int main(int argc, char* argv[])
{
	make_sure(argc==8,"argc<4,input graph path and *.ini");
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
	IntRandomUniform ir = IntRandomUniform(seed, 1, 100);
	seed = 1000;
	EdgeType userEdgeType = EdgeType::UNDEF_EDGE_TYPE;
	GraphRead* reader = getGraphReader(graphPath.c_str(), userEdgeType, ir);
	GraphWeight graphWeight(reader);
	graphWeight.name = fileUtil::extractFileName(graphPath);
	graphWeight.toCSR();
	graphWeight.analyseSimple();
	CudaConfigs configs = CudaConfigs(kv,vwSize,68*4,256,1024*8,dsl,ds);
	if(action == "run"){
		run(graphWeight,configs,testNodeSize);
	}
	else if(action == "test"){
		test(graphWeight,configs,testNodeSize);
	}else {
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
	vector<dist_t> dis(graph.v);
	double t, allt = 0.0, allkt = 0.0, allst = 0.0, allct = 0.0 ,allslt = 0.0;
	double allRN = 0.0, allRE = 0.0, allDP = 0.0 ,allRM = 0.0;

	if(!isCudaKv(configs.kernelVersion)){
		ComputeGraph* ct = getCtGraphFromConfig(graph, configs);
		for (int i = 0; i < testNodes.size(); i++) {
			ct->computeAndTick(testNodes[i], dis, t);
			cout << "Relax Source: " << testNodes[i] << "\trelaxNodes: " << 0 << "\trelaxEdges: " << 0 << "\tuseTime: " << t << endl;
			allt += t;
		}
	}
	else{
		ComputeGraph* cg = getCtGraphFromConfig(graph, configs);
		for (int i = 0; i < testNodes.size(); i++) {
			CudaProfiles pf = *(CudaProfiles*)cg->computeAndTick(testNodes[i], dis, t);
			if (printDeatil) {
				cout << "Relax Source: " << testNodes[i]
					<< "\trelaxNodes: " << pf.relaxNodes << "\trelaxNodesDivV: " << (double)pf.relaxNodes / graph.v
					<< "\trelaxEdges: " << pf.relaxEdges << "\trelaxEdgesDivE: " << (double)pf.relaxEdges / graph.e
					<< "\tdepth: " << pf.depth
					<< "\tuseTime: " << t
					<< "\tkernelTime: " << pf.kernel_time << "\tcacTime: " << pf.cac_time << "\tcopyTime: " << pf.copy_time 
					<< "\tselectTime: " << pf.select_time
					<< "\trelaxRemain: " << pf.relaxRemain
					<< endl;
			}
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
	cout << "\trelaxNodes: " << allRN << "\trelaxNodesDivV: " << allRN / graph.v
		<< "\trelaxEdges: " << allRE << "\trelaxEdgesDivE: " << allRE / graph.e
		<< "\trelaxDepth: " << allDP
		<< "\tuseTime: " << allt
		<< "\tkernelTime: " << allkt << "\tcacTime: " << allst << "\tcopyTime: " << allct
		<< "\tselectTime: " << allslt
		<< "\trelaxRemain: " << allRM
		<< endl;
}