#pragma once
#include <vector>
#include <vector_types.h>
#include <vector_functions.hpp>
using namespace std;

namespace graph {
	enum class	    EdgeType { DIRECTED, UNDIRECTED, UNDEF_EDGE_TYPE };
	enum class     GraphType { NORMAL, MULTIGRAPH, UNDEF_GRAPH_TYPE };
	enum class AttributeType { BINARY, INTEGER, REAL, SIGN };

	using node_t = int;
	using edge_t = int;
	using node_t2 = node_t[2];
	using degree_t = int;
	using dist_t = int;
	using weight_t = int;

	struct TriTuple {
		edge_t s, t;
		weight_t w;
		TriTuple(int a,int b,int c) {
			s = a, t = b, w = c;
		}
	};

	class GraphWeight {
	private:
		EdgeType edgeType;
		GraphType graphType;
		AttributeType attributeType;

		//vector<degree_t> inDegrees;
		//vector<degree_t> outDegrees;
		vector<TriTuple> originEdges;
	public:
		int v, e;
		vector<node_t> inNodes, outNodes;
		vector<edge_t> inEdges, outEdges;
		vector<weight_t> inWeights, outWeights;
		degree_t InDegree(int i) {
			return inNodes[i + 1] - inNodes[i];
		}
		degree_t OutDegree(int i) {
			return outNodes[i + 1] - outNodes[i];
		}

		GraphWeight(const int _V, const int _E, const EdgeType _edgeType, vector<TriTuple>& edges);
		~GraphWeight();
		void toCSR();
	};
}