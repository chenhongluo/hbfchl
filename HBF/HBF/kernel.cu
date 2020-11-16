#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

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
	//生成测试的点
	vector<int> testNodes(size, 0);
	IntRandomUniform iru(seed, 0, v);
	for (int i = 0; i < size; i++) {
		testNodes[i] = iru.getNextValue();
	}
	return testNodes;
}

int compareRes(vector<int>& res1, vector<int>& res2);
void test(GraphWeight &graph,boost::property_tree::ptree m_pt);
void run(GraphWeight &graph, boost::property_tree::ptree m_pt);
void predeal(GraphWeight &graph, boost::property_tree::ptree m_pt);

int main(int argc, char* argv[])
{
	// 读取参数配置data和*.config
	make_sure(argc==3,"argc!=2,input graph path and *.ini");
	string graphPath = argv[1];
	string ini = argv[2];
	if (!boost::filesystem::exists(graphPath)) {
		std::cerr << "graph not exists." << std::endl;
		return -1;
	}
	if (!boost::filesystem::exists(ini)) {
		std::cerr << "config.ini not exists." << std::endl;
		return -1;
	}
	boost::property_tree::ptree m_pt, tag_setting;
	try
	{
		read_ini(ini, m_pt);
	}
	catch (std::exception e)
	{
		__ERROR("config open error")
	}
	// 读取图相关的配置，生成相应的GraphWeight
	tag_setting = m_pt.get_child("graph");
	//Behind Camera Config ini
	string randomOpt = tag_setting.get<string>("random", "uniform");
	int seed = tag_setting.get<int>("seed", time(0));
	int random_min = tag_setting.get<int>("random_min", 1);
	int random_max = tag_setting.get<int>("random_max", 100);
	int edgeType = tag_setting.get<int>("edgeType", 100);
	bool compareFlag = tag_setting.get<bool>("compareFlag", false);

	EdgeType userEdgeType = EdgeType::UNDEF_EDGE_TYPE;
	if (edgeType == 1) {
		userEdgeType = EdgeType::DIRECTED;
	}
	else if (edgeType == 2) {
		userEdgeType = EdgeType::UNDIRECTED;
	}
	IntRandomUniform ir = IntRandomUniform();
	if (randomOpt == "uniform") {
		ir = IntRandomUniform(seed, random_min, random_max);
	}
	else {
		__ERROR("not exists random option")
	}
	GraphRead* reader = getGraphReader(graphPath.c_str(), userEdgeType, ir);
	GraphWeight graphWeight(reader);
	graphWeight.name = fileUtil::extractFileName(graphPath);
	graphWeight.toCSR();
	graphWeight.analyseSimple();
	tag_setting = m_pt.get_child("action");
	string action = tag_setting.get<string>("action");
	if (action == "run") {
		run(graphWeight, m_pt);
	}
	else if (action == "test") {
		test(graphWeight, m_pt);
	}
	else if (action == "predeal") {
		predeal(graphWeight, m_pt);
	}
	else if (action == "anaylse") {
		graphWeight.analyseDetail();
	}
    return 0;
}

CudaGraph* getCudaGraphFromConfig(GraphWeight &graph, boost::property_tree::ptree m_pt) {
	CudaConfigs configs;
	boost::property_tree::ptree tag_setting = m_pt.get_child("cuda");
	int gpuIndex = tag_setting.get<int>("gpu", 0);
	configs.gridDim = tag_setting.get<int>("gridDim", 278);
	configs.blockDim = tag_setting.get<int>("blockDim", 128);
	configs.sharedLimit = tag_setting.get<int>("sharedLimit", 1024);
	configs.kernelVersion = tag_setting.get<string>("kernel", "v0");
	configs.atomic64 = tag_setting.get<bool>("atomic64", true);
	configs.vwSize = tag_setting.get<int>("vwSize", true);
	cudaSetDevice(gpuIndex);
	return new CudaGraph (graph, configs);
}

CTHostGraph* getHostGraphFromConfig(GraphWeight &graph, boost::property_tree::ptree m_pt) {
	boost::property_tree::ptree tag_setting = m_pt.get_child("host");
	string kernel = tag_setting.get<string>("kernel");
	return CTHostGraph::getHostGraph(graph,kernel);
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

void test(GraphWeight &graph, boost::property_tree::ptree m_pt)
{
	boost::property_tree::ptree tag_setting;

	tag_setting = m_pt.get_child("action");
	string subaction = tag_setting.get<string>("subaction");
	int testNodeSize = tag_setting.get<int>("testNodeSize", 0);

	tag_setting = m_pt.get_child("cuda");
	CTHostGraph* cmpCt = CTHostGraph::getHostGraph(graph, "Dijkstra");
	vector<node_t> testNodes = getTestNodes(testNodeSize, 0, graph.v);
	vector<dist_t> dis1(graph.v), dis2(graph.v);
	double t1, t2;

	if (subaction == "host") {
		CTHostGraph* ct = getHostGraphFromConfig(graph, m_pt);
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
	else if (subaction == "cuda") {
		CudaGraph* cg = getCudaGraphFromConfig(graph, m_pt);
		for (int i = 0; i < testNodes.size(); i++) {
			cg->computeAndTick(testNodes[i], dis1, t1);
			cmpCt->computeAndTick(testNodes[i], dis2, t2);
			if (compareRes(dis1, dis2) == 0) {
				cout << "the " << i << " source node: " << testNodes[i] << " is correct" << endl;
			}
			else {
				cout << "the " << i << " source node: " << testNodes[i] << " is wrong" << endl;
			}
		}
	}
	else {
		__ERROR("no this subaction")
	}
}
void run(GraphWeight &graph, boost::property_tree::ptree m_pt)
{
	boost::property_tree::ptree tag_setting;

	tag_setting = m_pt.get_child("action");
	string subaction = tag_setting.get<string>("subaction");
	int testNodeSize = tag_setting.get<int>("testNodeSize", 0);

	tag_setting = m_pt.get_child("cuda");
	vector<node_t> testNodes = getTestNodes(testNodeSize, 0, graph.v);
	vector<dist_t> dis(graph.v);
	double t,allt = 0;
	double allRN = 0.0, allRE = 0.0;

	if (subaction == "host") {
		CTHostGraph* ct = getHostGraphFromConfig(graph, m_pt);
		for (int i = 0; i < testNodes.size(); i++) {
			ct->computeAndTick(testNodes[i], dis, t);
			cout << "Relax Source: " << testNodes[i] << "\trelaxNodes: " << 0 << "\trelaxEdges: " << 0 << "\tuseTime: " << t << endl;
			allt += t;
		}
	}
	else if (subaction == "cuda") {
		CudaGraph* cg = getCudaGraphFromConfig(graph, m_pt);
		for (int i = 0; i < testNodes.size(); i++) {
			CudaProfiles pf = cg->computeAndTick(testNodes[i], dis, t);
			cout << "Relax Source: " << testNodes[i]
				<< "\trelaxNodes: " << pf.relaxNodes << "\trelaxNodesDivV: " << (double)pf.relaxNodes / graph.v
				<< "\trelaxEdges: " << pf.relaxEdges << "\trelaxEdgesDivE: " << (double)pf.relaxEdges / graph.e
				<< "\tuseTime: " << t << endl;
			allt += t;
			allRN += pf.relaxNodes;
			allRE += pf.relaxEdges;
		}
	}
	else {
		__ERROR("no this subaction")
	}

	allRN /= testNodes.size();
	allRE /= testNodes.size();
	allt /= testNodes.size();

	cout << " Avg Profile: " << endl;
	cout << "\trelaxNodes: " << allRN << "\trelaxNodesDivV: " << allRN / graph.v
		<< "\trelaxEdges: " << allRE << "\trelaxEdgesDivE: " << allRE / graph.e
		<< "\tuseTime: " << allt << endl;
}
void predeal(GraphWeight &graph, boost::property_tree::ptree m_pt) 
{
	boost::property_tree::ptree tag_setting;

	tag_setting = m_pt.get_child("predeal");
	string kernel = tag_setting.get<string>("kernel");
	string dataDir = tag_setting.get<string>("dataDir");
	int n = tag_setting.get<int>("n",16);
	string filename;
	if (kernel == "none") {
		filename = dataDir + graph.name + ".gc";
	}
	else if (kernel == "preCompute") {
		filename = dataDir + graph.name + ".pc";
		filename += n;
		filename += ".gc";
		graph.preCompute(n);
	}
	else if (kernel == "reOrder") {
		filename = dataDir + graph.name + ".re.gc";
	}
	graph.toGC(filename.c_str());
}

