#include "graph.h"
#include <algorithm>
namespace graph {
	GraphWeight::~GraphWeight() {

	}

	vector<int2> GraphWeight::getOutEdgesOfNode(node_t v)
	{
		int s = outNodes[v];
		int t = outNodes[v + 1];
		return vector<int2>(&(outEdgeWeights[s]), &(outEdgeWeights[t]));
	}

	vector<int2> GraphWeight::getInEdgesOfNode(node_t v)
	{
		int s = inNodes[v];
		int t = inNodes[v + 1];
		return vector<int2>(&(inEdgeWeights[s]), &(inEdgeWeights[t]));
	}

	GraphWeight::GraphWeight(GraphRead* gr){
		this->reader = gr;
		GraphHeader header;
		cout<<"start reading graph"<<endl;
		reader->readData(header, originEdges);
		edgeType = header._edgeType;
		v = header._v;
		e = header._e;
	}
	GraphWeight::GraphWeight(const int _v, const int _e, const EdgeType _edgeType, vector<TriTuple>& edges) {
		originEdges = edges;
		edgeType = _edgeType;
		v = _v;
		e = _e;
	}
	void GraphWeight::toCSR() {
		printf("the graph start transfering to CSR format");
		outNodes.resize(v + 1, 0);
		outEdgeWeights.resize(e);

		if (edgeType == EdgeType::DIRECTED) {
			inNodes.resize(v + 1, 0);
			inEdgeWeights.resize(e);
		}
		vector<degree_t> outDegree(v, 0), outTemp(v, 0);
		for (TriTuple &tuple : originEdges) {
			outDegree[tuple.s]++;
		}
		for (int i = 1; i < v; i++) {
			outTemp[i] = outDegree[i - 1] + outTemp[i - 1];
			outNodes[i] = outTemp[i];
		}
		outNodes[v] = outNodes[v - 1] + outDegree[v - 1];
		for (TriTuple tuple : originEdges) {
			outEdgeWeights[outTemp[tuple.s]++] = make_int2(tuple.t, tuple.w);
		}
		if (edgeType == EdgeType::DIRECTED) {
			vector<degree_t> inDegree(v, 0), inTemp(v, 0);
			for (TriTuple &tuple : originEdges) {
				inDegree[tuple.t]++;
			}
			for (int i = 1; i < v; i++) {
				inTemp[i] = inDegree[i - 1] + inTemp[i - 1];
				inNodes[i] = inTemp[i];
			}
			inNodes[v] = inNodes[v - 1] + inDegree[v - 1];
			for (TriTuple tuple : originEdges) {
				inEdgeWeights[inTemp[tuple.t]++] = make_int2(tuple.s, tuple.w);
			}
		}
		else {
			inNodes = outNodes;
			inEdgeWeights = outEdgeWeights;
		}
	}
}