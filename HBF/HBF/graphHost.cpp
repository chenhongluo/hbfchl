#include "graph.h"
#include <queue>
#include <set>
#include <chrono>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>

namespace graph {

	dist_t * graph::GraphHost::BellmanFord_Result()
	{
		return distances;
	}

	void graph::GraphHost::bellmanFord_Queue_reset()
	{
		std::fill(distances, distances + v, std::numeric_limits<dist_t>::max());
	}

	void GraphHost::computeAndTick(node_t source,vector<dist_t>& res, double &t, HostSelector selector)
	{
		auto start = chrono::high_resolution_clock::now();
		switch (selector) {
		case HostSelector::BellmanFord:
			bellmanFord_Queue(source); break;
		case HostSelector::BellmanFordQueue:
			bellmanFord_Frontier(source); break;
		case HostSelector::Dijistra:
			dijkstraSET(source); break;
		case HostSelector::BoostBF:
			boostBellmanFord(source); break;
		case HostSelector::BoostD:
			boostDijkstra(source); break;
		}
		long long duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count();
		t = duration * 0.001;
		res.resize(v);
		for (int i = 0; i < v; i++) {
			res[i] = distances[i];
		}
	}

	namespace {
		bool relaxEdge(const node_t u, const node_t v,
			const weight_t weight, dist_t* d) {
			if (d[u] + weight < d[v]) {
				d[v] = d[u] + weight;
				return true;
			}
			return false;
		}
	}

	void graph::GraphHost::bellmanFord_Queue(const node_t source)
	{
		bellmanFord_Queue_reset();
		std::queue<node_t> q;
		q.push(source);
		distances[source] = 0;

		while (q.size() > 0) {
			const node_t next = q.front();
			q.pop();
			vector<int2> edges = graphWeight.getOutEdgesOfNode(next);
			for (int2 edge:edges) {
				if (relaxEdge(next, edge.x, edge.y, distances))
					q.push(edge.x);
			}
		}
	}

	void graph::GraphHost::bellmanFord_Frontier(const node_t source)
	{
		bellmanFord_Queue_reset();
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

	void graph::GraphHost::dijkstraSET(const node_t source)
	{
		bellmanFord_Queue_reset();

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
	void graph::GraphHost::boostDijkstra(const node_t source)
	{
		using namespace boost;

		typedef adjacency_list < listS, vecS, directedS, no_property, property<edge_weight_t, node_t> > graph_t;
		typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;

		graph_t g(edge_array, edge_array+e, weights, v);

		std::vector<vertex_descriptor> p(num_vertices(g));
		std::vector<int> d(num_vertices(g));
		vertex_descriptor s = vertex(source, g);

		dijkstra_shortest_paths(g, s,
			predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
			distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));

		int i = 0;
		graph_traits < graph_t >::vertex_iterator vi, vend;
		for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
			distances[i++] = d[*vi];
		}
	}



	void graph::GraphHost::boostBellmanFord(const node_t source)
	{
		using namespace boost;

		typedef std::pair < int, int > Edge;
		typedef adjacency_list < vecS, vecS, directedS, no_property, EdgeProperties> Graph;
		Graph g(edge_array, edge_array + e, v);

		graph_traits < Graph >::edge_iterator ei, ei_end;
		property_map<Graph, int EdgeProperties::*>::type weight_pmap = get(&EdgeProperties::weight, g);
		int i = 0;
		for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei, ++i)
			weight_pmap[*ei] = weights[i];

		std::vector<int> d(v, (std::numeric_limits < int >::max)());
		std::vector<std::size_t> parent(v);
		for (i = 0; i < v; ++i)
			parent[i] = i;
		d[source] = 0;

		bellman_ford_shortest_paths(g, int(v), weight_map(weight_pmap).distance_map(&d[0]).predecessor_map(&parent[0]));

		graph_traits < Graph >::vertex_iterator vi, vend;
		i = 0;
		for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
			distances[i++] = d[*vi];
		}
	}
}