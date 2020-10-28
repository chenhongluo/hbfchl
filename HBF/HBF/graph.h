#pragma once

#include <iostream>
#include <vector>
#include <vector_types.h>
#include <vector_functions.hpp>
#include "graphRead.h"
using namespace std;
namespace graph {
	class GraphRead;
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
		TriTuple() {
			s = t = w = -1;
		}
		TriTuple(int a,int b,int c) {
			s = a, t = b, w = c;
		}
		TriTuple(const TriTuple& tu) {
			s = tu.s;
			t = tu.t;
			w = tu.w;
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
		GraphRead* reader;
		vector<node_t> inNodes, outNodes;
		vector<int2> inEdgeWeights, outEdgeWeights;
		degree_t inDegree(int i) {
			return inNodes[i + 1] - inNodes[i];
		}
		degree_t outDegree(int i) {
			return outNodes[i + 1] - outNodes[i];
		}
		vector<int2> getOutEdgesOfNode(node_t v) const;
		vector<int2> getInEdgesOfNode(node_t v) const;

		GraphWeight(GraphRead* reader);
		GraphWeight(const int _V, const int _E, const EdgeType _edgeType, vector<TriTuple>& edges);
		~GraphWeight();
		void toCSR();
	};

	class GraphHost {
		const GraphWeight& graphWeight;
		dist_t* distances;
		const int v;
		const int e;
		typedef std::pair < int, int > Edge;
		Edge* edge_array;
		weight_t* weights;
	public:
		enum HostSelector { BellmanFord, BellmanFordQueue, Dijistra, BoostBF, BoostD };

		GraphHost(const GraphWeight& g) :graphWeight(g),v(g.v),e(g.e) {
			distances = new dist_t[v];
			edge_array = new Edge[e];
			weights = new weight_t[e];
			int k = 0;
			for (int i = 0; i < v; i++) {
				for (int j = graphWeight.outNodes[i]; j < graphWeight.outNodes[i + 1]; j++) {
					edge_array[k] = Edge(i, graphWeight.outEdgeWeights[j].x);
					weights[k] = graphWeight.outEdgeWeights[j].y;
					k = k + 1;
				}
			}
		}
		~GraphHost() { delete[] distances; delete[]edge_array; delete[]weights; }
		dist_t* BellmanFord_Result();
		void computeAndTick(node_t source, vector<dist_t> &res, double &t,HostSelector selector);
	private:
		void bellmanFord_Queue_reset();

		void bellmanFord_Queue(const node_t source);
		void bellmanFord_Frontier(const node_t source);
		void dijkstraSET(const node_t source);
		void boostDijkstra(const node_t source);
		void boostBellmanFord(const node_t source);
	};
	struct EdgeProperties {
		int weight;
	};
}