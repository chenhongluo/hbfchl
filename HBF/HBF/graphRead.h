#pragma once
#include<map>
#include <iostream>
#include <fstream>
#include <string>
#include "graph.h"
#include "fUtil.h"
using namespace std;
using namespace randomUtil;
namespace graph {
	enum class	    EdgeType;
	enum class     GraphType;
	enum class AttributeType;
	struct TriTuple;
	struct ShortCut;
	class GraphHeader {
	public:
		int _v;
		int _e;
		EdgeType _edgeType;
		GraphHeader() {}
	};
	class GraphRead {
	public:
		void readData(GraphHeader& header, vector<TriTuple>& edges);
		virtual vector<unsigned> getOrders() { return {}; }
		virtual vector<ShortCut> getAddEdges() { return {}; }
		GraphRead(const char * filename, EdgeType direction, IntRandom& ir);
		virtual ~GraphRead();
	protected:
		bool fileWeightFlag;
		bool symmeticFlag;
		int v;
		int e;
		int nof_lines;
		ifstream fin;
		string filename;
		EdgeType userDirection;
		EdgeType fileDirection;
		IntRandom& weightRandom;
		GraphHeader makeSureDirection();
		virtual GraphHeader getHeader() = 0;
		virtual vector<TriTuple> getOriginalEdegs() = 0;
	};

	class grReader: public GraphRead {
	public:
		grReader(const char * filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
	};

	class MatrixMarketReader :public GraphRead {
	public:
		MatrixMarketReader(const char * filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
	};

	class Dimacs10Reader :public GraphRead {
	public:
		Dimacs10Reader(const char * filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
	};
	class Dimacs9Reader :public GraphRead {
	public:
		Dimacs9Reader(const char * filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
	};
	class SnapReader :public GraphRead {
	public:
		SnapReader(const char * filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
	};

	class DDSGReader :public GraphRead {
	public:
		DDSGReader(const char *filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
	};

	class CHReader :public GraphRead {
	public:
		CHReader(const char *filename, EdgeType direction, IntRandom& ir);
		GraphHeader getHeader();
		vector<TriTuple> getOriginalEdegs();
		vector<unsigned> getOrders();
		vector<ShortCut> getAddEdges();
	};

	GraphRead * getGraphReader(const char * filename, EdgeType direction, randomUtil::IntRandom& ir);
}