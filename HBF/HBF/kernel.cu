#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <stdio.h>
#include "graph.h"
#include "cudaGraph.cuh"
using namespace graph;
using namespace cuda_graph;

void test() 
{
	vector<TriTuple> ts;
	ts.push_back(TriTuple(0, 2, 5));
	ts.push_back(TriTuple(1, 2, 1));
	ts.push_back(TriTuple(1, 3, 1));
	ts.push_back(TriTuple(1, 4, 1));
	ts.push_back(TriTuple(2, 4, 2));
	ts.push_back(TriTuple(2, 3, 2));
	ts.push_back(TriTuple(3, 4, 3));
	ts.push_back(TriTuple(4, 1, 4));
	GraphWeight hostGraph(5, 8, EdgeType::UNDIRECTED, ts);
	//EdgeType userEdgeType = EdgeType::UNDEF_EDGE_TYPE;
	//IntRandomUniform ir = IntRandomUniform();
	//GraphRead* reader = getGraphReader("F:/data_graph/delaunay_n20.graph", userEdgeType,ir);
	//GraphWeight hostGraph = GraphWeight(reader);
	//vector<int2> v1 = hostGraph.getOutEdgesOfNode(0);
	//vector<int2> v2 = hostGraph.getInEdgesOfNode(186869);
	hostGraph.toCSR();
	GraphHost hg(hostGraph);
	vector<dist_t> res1, res2;
	double t1, t2;
	hg.computeAndTick(0, res1, t1, GraphHost::HostSelector::BoostD);
	hg.computeAndTick(0, res2, t2, GraphHost::HostSelector::BellmanFord);
}
#define make_sure(expression,msg)\
do{\
if(!(expression))\
{\
	__ERROR(msg) \
}\
}while(0)

int compareRes(vector<int>& res1, vector<int>& res2);

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
	tag_setting = m_pt.get_child("cuda");
	CudaConfigs configs;

	int gpuIndex = tag_setting.get<int>("gpu", 0);
	cudaSetDevice(gpuIndex);
	configs.gridDim = tag_setting.get<int>("gridDim", 278);
	configs.blockDim = tag_setting.get<int>("blockDim", 128);
	configs.sharedLimit = tag_setting.get<int>("sharedLimit", 1024);
	configs.kernelVersion = tag_setting.get<string>("kernel", "v0");
	configs.atomic64 = tag_setting.get<bool>("atomic64", true);
	configs.vwSize = tag_setting.get<int>("vwSize", true);


	tag_setting = m_pt.get_child("host");
	//Behind Camera Config ini
	string randomOpt= tag_setting.get<string>("random", "uniform");
	int seed = tag_setting.get<int>("seed", time(0));
	int random_min = tag_setting.get<int>("random_min", 1);
	int random_max = tag_setting.get<int>("random_max", 100);
	int testNodeSize = tag_setting.get<int>("testNodeSize", 100);
	bool compareFlag = tag_setting.get<bool>("compareFlag", false);
	int edgeType = tag_setting.get<int>("testNodes", 0);

	EdgeType userEdgeType = EdgeType::UNDEF_EDGE_TYPE;
	if (edgeType == 1) {
		userEdgeType = EdgeType::DIRECTED;
	}
	else if (edgeType == 2) {
		userEdgeType = EdgeType::UNDIRECTED;
	}
	IntRandomUniform ir = IntRandomUniform();
	if(randomOpt == "uniform"){
		ir = IntRandomUniform(seed, random_min,random_max);
	}
	else {
		__ERROR("not exists random option")
	}
	GraphRead* reader = getGraphReader(graphPath.c_str(), userEdgeType,ir);
	GraphWeight graphWeight(reader);
	graphWeight.toCSR();
	//vector<int2> v1 = hostGraph.getOutEdgesOfNode(0);
	//vector<int2> v2 = hostGraph.getInEdgesOfNode(186869);
	GraphHost hg(graphWeight);
	CudaGraph cg(graphWeight, configs);
	// 构建图
	// 生成GraphHost和CudaGraph

	//生成测试的点
	vector<int> testNodes(testNodeSize, 0);
	IntRandomUniform iru(seed, 0, graphWeight.v);
	for (int i = 0; i < testNodeSize; i++) {
		testNodes[i] = iru.getNextValue();
	}

	for (int i = 0; i < testNodeSize; i++) {
		int source = testNodes[i];
		cout << "source = " << source << endl;
		double t1,t2;
		vector<int> cudaRes, hostRes;
		cg.computeAndTick(source,cudaRes,t1);
		//hg.computeAndTick(source, cudaRes, t1, GraphHost::HostSelector::BoostD);
		if (compareFlag) {
			hg.computeAndTick(source, hostRes, t2, GraphHost::HostSelector::Dijistra);
			int flag = compareRes(hostRes, cudaRes);
			if (flag == -2)
				__ERROR("compareRes flag==-2")
			else if (flag == -1)
				__ERROR("compareRes flag==-1")
			else
				cout << "pass test " << i << endl;
		}
	}
	// 完成后生成source nodes开始search
	// 取得结果后对比 指标采集是下一步的任务
    return 0;
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