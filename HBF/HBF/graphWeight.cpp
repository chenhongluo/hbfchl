#include "graph.h"
#include <set>
#include <queue>
#include <algorithm>
namespace graph {
	GraphWeight::~GraphWeight() {

	}

	vector<int2> GraphWeight::getOutEdgesOfNode(node_t v) const
	{
		int s = outNodes[v];
		int t = outNodes[v + 1];
		const int2* p = &outEdgeWeights[0];
		return vector<int2>(p + s, p + t);
	}

	vector<int2> GraphWeight::getInEdgesOfNode(node_t v) const
	{
		int s = inNodes[v];
		int t = inNodes[v + 1];
		return vector<int2>(&(inEdgeWeights[s]), &(inEdgeWeights[t]));
	}

	void GraphWeight::toDDSG(const char * filename)
	{
		ofstream of(filename);
		of << "d" << endl;
		of << v << " " << e << endl;
		for (int i = 0; i < e; i++) {
			of << originEdges[i].s <<" "<< originEdges[i].t << " " << originEdges[i].w << " " << 1 << endl;
		}
		of.close();
	}

	//weight_t getDistance(map<node_t, weight_t> m, node_t v) {
	//	if (m.find(v) == m.end()) {
	//		return INT_MAX;
	//	}
	//	else
	//	{
	//		return m[v];
	//	}
	//}
	//void GraphWeight::preCompute(int n)
	//{
	//	cout << "preComputing..." << endl;
	//	typedef std::pair<weight_t, node_t> Distance;
	//	vector<TriTuple> newOriginalEdges;
	//	newOriginalEdges.reserve(e + v * n);
	//	fUtil::Progress progress(v);
	//	int update = 0;
	//	for (int i = 0; i < v; i++) {
	//		progress.next(i + 1);
	//		std::set<Distance> actives;
	//		std::map<node_t,weight_t> distances;
	//		std::map<node_t, weight_t> results;
	//		distances[i] = 0;
	//		actives.insert(Distance(0, i));
	//		//int iter = (outNodes[i + 1] - outNodes[i]) * 2;
	//		int iter = n;
	//		for (int k = 0; k < iter; k++) {
	//			if (actives.size() == 0)
	//				break;
	//			Distance dis = *actives.begin();
	//			results[dis.second] = dis.first;
	//			actives.erase(actives.begin());

	//			const node_t rn = dis.second;
	//			const weight_t rw = dis.first;
	//			for (int j = outNodes[rn]; j < outNodes[rn + 1]; j++) {
	//				const int2 destxy = outEdgeWeights[j];
	//				const node_t dest = destxy.x;
	//				const weight_t oldweight = getDistance(distances, dest);
	//				const weight_t newweight = rw + destxy.y;
	//				if (newweight < oldweight) {
	//					if (oldweight != INT_MAX) {
	//						actives.erase(Distance(oldweight, dest));
	//						update++;
	//					}
	//					actives.insert(Distance(newweight, dest));
	//					distances[dest] = newweight;
	//				}
	//			}
	//		}
	//		map<node_t, weight_t> newEdges;
	//		for (auto &x : getOutEdgesOfNode(i)) {
	//			if (getDistance(newEdges, x.x) > x.y)
	//				newEdges[x.x] = x.y;
	//		}
	//		for (auto &x : results) {
	//			node_t dest = x.first;
	//			weight_t weight = x.second;
	//			newEdges[dest] = weight;
	//		}
	//		//for (auto &x : newEdges) {
	//		//	newOriginalEdges.push_back(TriTuple(i, x.first, x.second));
	//		//}
	//	}
	//	originEdges = newOriginalEdges;
	//	cout << "end preCompute: " << update << endl;
	//}

	double getDis(vector<int> array) {
		double sum = 0.0;
		for (auto &x : array) { sum += x; }
		double mean = sum / array.size(); //¾ùÖµ

		double accum = 0.0;
		std::for_each(std::begin(array), std::end(array), [&](const double d) {
			accum += (d - mean)*(d - mean);
		});

		double stdev = sqrt(accum / array.size() ); //·½²î
		return stdev;
	}

	void GraphWeight::analyseDetail()
	{
		cout << "graphName: " << name << endl;

		vector<int> ds(v);
		for (int i = 0; i < v; i++) {
			ds[i] = outNodes[i + 1] - outNodes[i];
		}
		fUtil::analyseIntVec<true>(ds, "degree detail anaylse");

		/*vector<int> edges(e);
		for (int i = 0; i < e; i++) {
			edges[i] = outEdgeWeights[i].y;
		}
		fUtil::analyseIntVec<true>(edges, 10000, "edge weight detail anaylse");*/

		/*cout << " locality detail anaylse" << endl;
		int k = 0;
		double dis = 0.0;
		int yield = 8;
		int yieldk = 3;
		for (int i = 0; i < v; i += 1) {
			vector<node_t> nodes1,nodes2;
			nodes1.push_back(i);
			int deep = 0;
			while (nodes1.size() > 0 && nodes1.size() < yield && deep < yieldk) {
				for (auto x : nodes1) {
					for (auto y : getOutEdgesOfNode(x)) {
						nodes2.push_back(y.x);
					}
				}
				swap(nodes1, nodes2);
				nodes2.clear();
				deep++;
			}
			if (nodes1.size() > 0) {
				k = k + 1;
				dis = (dis * (k - 1)) / k + getDis(nodes1) / k;
			}
			if (i % 1 == 0) {
				cout << "now i = "<< i << " temp locality value: " << dis << endl;
			}
		}
		cout << "locality yield: " << yield << "\tlocality value: " << dis << endl;*/
	}

	void GraphWeight::analyseSimple()
	{
		cout << "graphName: " << name;
		if (addFlag != 0)
			cout << addUsePercent;
		cout << "\tV: " << v << "\tE: " << e << "\tavgDegree: " << (double)e / v << endl;
	}

	GraphWeight::GraphWeight(GraphRead* gr){
		this->reader = gr;
		GraphHeader header;
		cout<<"start reading graph"<<endl;
		reader->readData(header, originEdges);
		edgeType = header._edgeType;
		v = header._v;
		e = header._e;
		orders = reader->getOrders();
		addEdges = reader->getAddEdges();
	}
	GraphWeight::GraphWeight(const int _v, const int _e, const EdgeType _edgeType, vector<TriTuple>& edges) {
		originEdges = edges;
		edgeType = _edgeType;
		v = _v;
		e = _e;
	}
	void GraphWeight::toCSR() {
		cout << "the graph start transfering to CSR format" << endl;
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