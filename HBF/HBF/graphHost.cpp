#include "graph.h"
#include <queue>
#include <set>
#include <chrono>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>

namespace graph {

	void CTHostGraph::bellmanFord_Queue_reset()
	{
		std::fill(distances.begin(), distances.end(), std::numeric_limits<dist_t>::max());
	}

	void CTHostGraph::computeAndTick(node_t source, vector<dist_t>& res, double & t)
	{
		auto start = chrono::high_resolution_clock::now();
		bellmanFord_Queue_reset();
		compute(source);
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		res.resize(v);
		for (int i = 0; i < v; i++) {
			res[i] = distances[i];
		}
		//cout << "Relax Source: " << source << "\trelaxNodes: " << 0 << "\trelaxEdges: " << 0 << "\tuseTime: " << t << endl;
	}

	CTHostGraph* CTHostGraph::getHostGraph(const GraphWeight& hg, string selector)
	{
		if (selector == "BellmanFord")
			return new BFHostGraph(hg);
		else if (selector == "Dijkstra")
			return new DJHostGraph(hg);
		else if (selector == "BoostBellmanFord")
			return new BBLHostGraph(hg);
		else if (selector == "BoostDijstra")
			return new BDJHostGraph(hg);
		else
		{
			__ERROR("host no this kernel")
		}
	}

	void DJHostGraph::compute(node_t source)
	{
		typedef std::pair<weight_t, node_t> Node;
		std::set<Node> PriorityQueue;
		PriorityQueue.insert(Node(0, source));

		distances[source] = 0;
		while (!PriorityQueue.empty()) {
			const node_t next = PriorityQueue.begin()->second;
			PriorityQueue.erase(PriorityQueue.begin());

			for (int j = graphWeight.outNodes[next]; j < graphWeight.outNodes[next + 1]; j++) {
				const int2 destxy = graphWeight.outEdgeWeights[j];
				const node_t dest = destxy.x;
				const weight_t w = destxy.y;
				if (distances[next] + w < distances[dest]) {
					PriorityQueue.erase(Node(distances[dest], dest));
					distances[dest] = distances[next] + w;
					PriorityQueue.insert(Node(distances[dest], dest));
				}
			}
		}
	}

	namespace {
		bool relaxEdge(const node_t u, const node_t v,
			const weight_t weight,vector<dist_t> d) {
			if (d[u] + weight < d[v]) {
				d[v] = d[u] + weight;
				return true;
			}
			return false;
		}
	}

	void BFHostGraph::compute(node_t source)
	{
		std::queue<node_t> q1;
		std::queue<node_t> q2;
		q1.push(source);
		distances[source] = 0;
		int level = 0, visited = 0;
		while (q1.size() > 0) {
			while (q1.size() > 0) {
				const node_t next = q1.front();
				q1.pop();
				vector<int2> edges = graphWeight.getOutEdgesOfNode(next);
				for (int2 edge : edges) {
					if (relaxEdge(next, edge.x, edge.y, distances))
						q2.push(edge.x);
				}
			}
			swap(q1, q2);
		}
	}
	void BDJHostGraph::compute(node_t source) {}
	void BDJHostGraph::computeAndTick(node_t source, vector<dist_t>& res, double &t)
	{
		using namespace boost;
		typedef adjacency_list < boost::listS, boost::vecS, boost::directedS, boost::no_property, property<edge_weight_t, node_t> > graph_t;
		typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
		typedef std::pair<node_t, node_t> Edge;
		graph_t g(edge_array, &edge_array[e], weights, v);
		std::vector<vertex_descriptor> p(num_vertices(g));
		std::vector<int> d(num_vertices(g));
		vertex_descriptor s = vertex(source, g);
		auto start = chrono::high_resolution_clock::now();
		dijkstra_shortest_paths(g, s,
			predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
			distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		int i = 0;
		res.resize(v);
		graph_traits < graph_t >::vertex_iterator vi, vend;
		for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
			res[i++] = d[*vi];
		}
	}

	struct EdgeProperties {
		int weight;
	};

	void BBLHostGraph::compute(node_t source) {}
	void BBLHostGraph::computeAndTick(node_t source, vector<dist_t>& res, double &t)
	{
		using namespace boost;
		typedef std::pair < int, int > Edge;
		typedef adjacency_list < vecS, vecS, directedS, no_property, EdgeProperties> graph_t;

		graph_t g(edge_array, edge_array + e, v);

		graph_traits < graph_t >::edge_iterator ei, ei_end;
		property_map<graph_t, int EdgeProperties::*>::type weight_pmap = get(&EdgeProperties::weight, g);
		int i = 0;
		for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei, ++i)
			weight_pmap[*ei] = weights[i];
		std::vector<int> d(v, (std::numeric_limits < int >::max)());
		std::vector<std::size_t> parent(v);
		auto start = chrono::high_resolution_clock::now();
		for (i = 0; i < v; ++i)
			parent[i] = i;
		d[source] = 0;
		bellman_ford_shortest_paths(g, int(v), weight_map(weight_pmap).distance_map(&d[0]).predecessor_map(&parent[0]));
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		res.resize(v);
		graph_traits < graph_t >::vertex_iterator vi, vend;
		for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
			res[i++] = d[*vi];
		}
	}
}