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
		GraphRead(const char * filename, EdgeType direction, IntRandom& ir);
		virtual ~GraphRead();
	protected:
		bool fileWeightFlag;
		bool symmeticFlag;
		int v;
		int e;
		int nof_lines;
		ifstream fin;
		EdgeType userDirection;
		EdgeType fileDirection;
		IntRandom& weightRandom;
		GraphHeader makeSureDirection();
		virtual GraphHeader getHeader() = 0;
		virtual vector<TriTuple> getOriginalEdegs() = 0;
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

	GraphRead * getGraphReader(const char * filename, EdgeType direction, randomUtil::IntRandom& ir);
}