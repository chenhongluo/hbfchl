#include "graph.h"
#include <algorithm>
namespace {

}
namespace graph {
	GraphWeight::GraphWeight(const int _v, const int _e, const EdgeType _edgeType, vector<TriTuple>& edges) {
		originEdges = edges;
		edgeType = _edgeType;
		v = _v;
		e = _e;
		outNodes.resize(v + 1,0);
		outEdges.resize(e);
		outWeights.resize(e);

		if (edgeType == EdgeType::DIRECTED) {
			inNodes.resize(v + 1,0);
			inEdges.resize(e);
			inWeights.resize(e);
		}
		vector<degree_t> outDegree(v,0), outTemp(v, 0);
		for (TriTuple &tuple : edges) {
			outDegree[tuple.s]++;
		}
		for (int i = 1; i < v; i++) {
			outTemp[i] = outDegree[i - 1] + outTemp[i - 1];
			outNodes[i] = outTemp[i];
		}
		outNodes[v] = outNodes[v - 1] + outDegree[v - 1];
		for (TriTuple tuple : edges) {
			outEdges[outTemp[tuple.s]] = tuple.t;
			outWeights[outTemp[tuple.s]++] = tuple.w;
		}
		if (edgeType == EdgeType::DIRECTED) {
			vector<degree_t> inDegree(v, 0), inTemp(v, 0);
			for (TriTuple &tuple : edges) {
				inDegree[tuple.t]++;
			}
			for (int i = 1; i < v; i++) {
				inTemp[i] = inDegree[i - 1] + inTemp[i - 1];
				inNodes[i] = inTemp[i];
			}
			inNodes[v] = inNodes[v - 1] + inDegree[v - 1];
			for (TriTuple tuple : edges) {
				inEdges[inTemp[tuple.t]] = tuple.s;
				inWeights[inTemp[tuple.t]++] = tuple.w;
			}
		}
		else {
			inNodes = (vector<node_t>&)outNodes;
			inEdges = (vector<edge_t>&)outEdges;
			inWeights = (vector<weight_t>&) outWeights;
		}
	}
	GraphWeight::~GraphWeight() {

	}
}