#include<map>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <iterator>
#include<sstream>

#include"graphRead.h"

using namespace std;
namespace graph {
	void GraphRead::readData(GraphHeader & header, vector<TriTuple>& edges)
	{
		fin.seekg(std::ios::beg);
		header = getHeader();
		fin.seekg(std::ios::beg);
		edges = getOriginalEdegs();
		header._e = edges.size();
	}
	GraphRead::GraphRead(const char* filename, EdgeType direction, randomUtil::IntRandom& ir):weightRandom(ir) {
		fileUtil::checkRegularFile(filename);
		long long int size = fileUtil::fileSize(filename);
		symmeticFlag = false;
		fileWeightFlag = false;
		StreamModifier::thousandSep();

		std::cout << std::endl << "Read File:\t" << fileUtil::extractFileName(filename) << "\tSize: " << size / (1024 * 1024) << " MB" << std::endl;
		//fin >> flag;
		//fin.seekg(std::ios::beg);
		this->filename = filename;
		fin = ifstream(filename);
		userDirection = direction;
		fileDirection = EdgeType::UNDEF_EDGE_TYPE;
	}

	GraphRead::~GraphRead()
	{
		fin.close();
	}

	GraphRead* getGraphReader(const char* filename, EdgeType direction, IntRandom& ir)
	{
		string fileExtension = fileUtil::extractFileExtension(string(filename));
		if (fileExtension == ".ddsg") {
			return new DDSGReader(filename, direction, ir);
		}
		if (fileExtension == ".cn") {
			return new CHReader(filename, direction, ir);
		}

		ifstream f = ifstream(filename);
		string s;
		f >> s;
		f.close();
		if (s.compare("c") == 0 || s.compare("p") == 0)
			return new Dimacs9Reader(filename, direction, ir);
		else if (s.compare("%%MatrixMarket") == 0)
			return new MatrixMarketReader(filename, direction, ir);
		else if (s.compare("#") == 0)
			return new SnapReader(filename, direction, ir);
		else if (s.compare("%") == 0 || fUtil::isDigit(s.c_str()))
			return new Dimacs10Reader(filename, direction, ir);
		else
			__ERROR(" Error. Graph Type not recognized: " << filename << " " << s)
	}

	GraphHeader GraphRead::makeSureDirection()
	{
		GraphHeader header;
		bool undirectedFlag = userDirection == EdgeType::UNDIRECTED || (userDirection == EdgeType::UNDEF_EDGE_TYPE && fileDirection == EdgeType::UNDIRECTED);

		std::string graphDir = undirectedFlag ? "GraphType: Undirected " : "GraphType: Directed ";
		header._v = v;
		header._e = e = undirectedFlag ? nof_lines * 2 : nof_lines;
		header._edgeType = (undirectedFlag == true) ? EdgeType::UNDIRECTED : EdgeType::DIRECTED;
		if (userDirection != EdgeType::UNDEF_EDGE_TYPE)
			graphDir.append("(User Def)");
		else if (fileDirection != EdgeType::UNDEF_EDGE_TYPE) {
			graphDir.append("(File Def)");
			userDirection = fileDirection;
		}
		else {
			graphDir.append("(UnDef)");
			userDirection = EdgeType::DIRECTED;
		}
		std::cout << std::endl << "\tNodes: " << header._v << "\tEdges: " << header._e << "\tSymmetricFlag:" << symmeticFlag
			<< "\tFileWeightFlag:" << fileWeightFlag << '\t' << graphDir
			<< "\tDegree AVG: " << std::fixed << std::setprecision(1) << (float)header._e / header._v << std::endl;
		StreamModifier::resetSep();
		return header;
	}


	MatrixMarketReader::MatrixMarketReader(const char* filename, EdgeType direction, randomUtil::IntRandom& ir)
		:GraphRead(filename,direction,ir)
	{
	}

	GraphHeader MatrixMarketReader::getHeader(){
		std::string MMline;
		
		std::getline(fin, MMline);
		symmeticFlag = (MMline.find("symmetric") != std::string::npos);
		fileDirection = symmeticFlag ? EdgeType::UNDIRECTED : EdgeType::DIRECTED;
		/*if (MMline.find("real") != std::string::npos)
			FileAttributeType = AttributeType::REAL;
		else if (MMline.find("integer") != std::string::npos)
			FileAttributeType = AttributeType::INTEGER;
		else
			FileAttributeType = AttributeType::BINARY;*/
		while (fin.peek() == '%')
			fileUtil::skipLines(fin);

		fin >> v >> MMline >> nof_lines;
		return makeSureDirection();
	}

	vector<TriTuple> MatrixMarketReader::getOriginalEdegs() {
		fUtil::Progress progress(nof_lines);

		while (fin.peek() == '%')
			fileUtil::skipLines(fin);
		fileUtil::skipLines(fin);
		vector<TriTuple> originEdges;
		originEdges.reserve(e);
		for (int lines = 0; lines < nof_lines; ++lines) {
			node_t index1, index2;
			fin >> index1 >> index2;
			int weight = weightRandom.getNextValue();
			originEdges.push_back(TriTuple(index1 - 1, index2 - 1, weight));
			if(userDirection == EdgeType::UNDIRECTED)
				originEdges.push_back(TriTuple(index2 - 1, index1 - 1, weight));
			progress.next(lines + 1);
			fileUtil::skipLines(fin);
		}
		return std::move(originEdges);
	}


	Dimacs10Reader::Dimacs10Reader(const char* filename, EdgeType direction, randomUtil::IntRandom& ir)
		:GraphRead(filename, direction, ir)
	{
	}

	GraphHeader Dimacs10Reader::getHeader()
	{
		while (fin.peek() == '%')
			fileUtil::skipLines(fin);

		std::string str;
		fin >> v >> nof_lines >> str;
		fileDirection = str.compare("100") == 0 ? EdgeType::DIRECTED : EdgeType::UNDIRECTED;
		if (fileDirection == EdgeType::UNDIRECTED)
		{
			symmeticFlag = true;
			nof_lines *= 2;
			fileDirection = EdgeType::DIRECTED;
		}
		return makeSureDirection();
	}

	vector<TriTuple> Dimacs10Reader::getOriginalEdegs()
	{
		fUtil::Progress progress(v);
		while (fin.peek() == '%')
			fileUtil::skipLines(fin);
		fileUtil::skipLines(fin);
		vector<TriTuple> originEdges;
		originEdges.reserve(e);
		for (int lines = 0; lines < v; lines++) {
			std::string str;
			std::getline(fin, str);

			std::istringstream stream(str);
			std::istream_iterator<std::string> iis(stream >> std::ws);

			degree_t degree = std::distance(iis, std::istream_iterator<std::string>());
			std::istringstream stream2(str);
			for (int j = 0; j < degree; j++) {
				node_t dest;
				stream2 >> dest;
				dest--;
				originEdges.push_back(TriTuple(lines,dest,weightRandom.getNextValue()));
				// problem TODO
			}
			progress.next(lines + 1);
		}
		return std::move(originEdges);
	}


	Dimacs9Reader::Dimacs9Reader(const char * filename, EdgeType direction, IntRandom& ir):GraphRead(filename,direction,ir)
	{
	}

	GraphHeader Dimacs9Reader::getHeader()
	{
		while (fin.peek() == 'c')
			fileUtil::skipLines(fin);

		std::string nil;
		fin >> nil >> nil >> v >> nof_lines;
		if (userDirection == EdgeType::UNDIRECTED) {
			symmeticFlag = true;
		}
		fileWeightFlag = true;
		return makeSureDirection();
	}

	vector<TriTuple> Dimacs9Reader::getOriginalEdegs()
	{
		fUtil::Progress progress(nof_lines);

		char c;
		int lines = 0;
		std::string nil;
		vector<TriTuple> originEdges;
		originEdges.reserve(e);
		while ((c = fin.peek()) != EOF) {
			if (c == 'a') {
				node_t index1, index2;
				weight_t weight;
				fin >> nil >> index1 >> index2 >> weight;
				originEdges.push_back(TriTuple(index1 - 1, index2 - 1, weight));
				lines++;
				progress.next(lines + 1);
			}
			fileUtil::skipLines(fin);
		}
		return std::move(originEdges);
	}


	SnapReader::SnapReader(const char * filename, EdgeType direction, IntRandom& ir):GraphRead(filename,direction,ir)
	{
	}

	GraphHeader SnapReader::getHeader()
	{
		std::string tmp;
		fin >> tmp >> tmp;
		symmeticFlag = tmp.compare("Undirected") == 0;
		fileDirection = symmeticFlag ? EdgeType::UNDIRECTED : EdgeType::DIRECTED;
		fileUtil::skipLines(fin);

		while (fin.peek() == '#') {
			std::getline(fin, tmp);
			if (tmp.substr(2, 6).compare("Nodes:") == 0) {
				std::istringstream stream(tmp);
				stream >> tmp >> tmp >> v >> tmp >> nof_lines;
				break;
			}
		}
		fileUtil::skipLines(fin);
		std::string MMline;
		std::getline(fin, MMline);
		//FileAttributeType = MMline.find("Sign") != std::string::npos ?
		//                        AttributeType::SIGN :: AttributeType::BINARY;
		return makeSureDirection();
	}

	namespace util{
		node_t getNodeId(unordered_map<node_t, node_t> &m, node_t sid) {
			auto it = m.find(sid);
			if (it == m.end()) {
				node_t id = m.size();
				m.insert(make_pair(sid,id));
			}
			return m[sid];
		}
	}

	vector<TriTuple> SnapReader::getOriginalEdegs()
	{
		fUtil::Progress progress(nof_lines);
		while (fin.peek() == '#')
			fileUtil::skipLines(fin);

		vector<TriTuple> originEdges;
		originEdges.reserve(e);
		unordered_map<node_t, node_t> m;
		for (int lines = 0; lines < nof_lines; lines++) {
			node_t ID1, ID2;
			fin >> ID1 >> ID2;
			originEdges.push_back(TriTuple(util::getNodeId(m, ID1), util::getNodeId(m, ID2), weightRandom.getNextValue()));
			progress.next(lines + 1);
		}
		return std::move(originEdges);
	}


	DDSGReader::DDSGReader(const char * filename, EdgeType direction, IntRandom & ir):GraphRead(filename, direction, ir)
	{
		userDirection = EdgeType::UNDEF_EDGE_TYPE;
		fileDirection = EdgeType::DIRECTED;
		fileWeightFlag = true;
	}

	GraphHeader DDSGReader::getHeader()
	{
		while (fin.peek() == 'd')
			fileUtil::skipLines(fin);
		fin >> v >> nof_lines;
		fileWeightFlag = true;
		return makeSureDirection();
	}

	vector<TriTuple> DDSGReader::getOriginalEdegs()
	{
		while (fin.peek() == 'd')
			fileUtil::skipLines(fin);
		fileUtil::skipLines(fin);

		vector<TriTuple> originEdges;
		originEdges.reserve(2 * e);
		fUtil::Progress progress(nof_lines);
		for (int lines = 0; lines < nof_lines; lines++) {
			node_t x,y;
			weight_t w;
			int flag;
			fin >> x >> y >> w >> flag;
			if (flag == 0 || flag == 1 || flag == 3) {
				originEdges.push_back(TriTuple(x, y, w));
			}
			if (flag == 0 || flag == 2 || flag == 3) {
				originEdges.push_back(TriTuple(y, x, w));
			}
			progress.next(lines + 1);
		}
		originEdges.shrink_to_fit();
		e = originEdges.size();
		return originEdges;
	}

	CHReader::CHReader(const char * filename, EdgeType direction, IntRandom & ir) : GraphRead(filename, direction, ir)
	{
		userDirection = EdgeType::UNDEF_EDGE_TYPE;
		fileDirection = EdgeType::DIRECTED;
		fileWeightFlag = true;
	}

	GraphHeader CHReader::getHeader()
	{
		ifstream bfin(filename, ios::in | ios::binary);
		unsigned value;
		bfin.read((char*)&value, sizeof(value)); //CH\r\n
		bfin.read((char*)&value, sizeof(value)); // 1
		if (value != 1) {
			__ERROR("uncorrect CH version")
		}
		bfin.read((char*)&v, sizeof(value)); // v
		bfin.read((char*)&nof_lines, sizeof(value)); // m1
		bfin.close();
		return makeSureDirection();
	}

	vector<TriTuple> CHReader::getOriginalEdegs()
	{
		ifstream bfin(filename, ios::in | ios::binary);

		vector<TriTuple> originEdges;
		originEdges.reserve(e);
		fUtil::Progress progress(nof_lines);
		bfin.seekg((5 + v) * 4, ios::beg); // skip header and order
		for (int lines = 0; lines < nof_lines; lines++) {
			node_t x, y;
			weight_t w;
			int flag;
			bfin.read((char*)&x, sizeof(node_t));
			bfin.read((char*)&y, sizeof(node_t));
			bfin.read((char*)&w, sizeof(node_t));
			bfin.read((char*)&flag, sizeof(node_t));
			if (flag != 1 && flag != 2 && flag != 3) {
				__ERROR("flag uncorrect")
			}
			if (flag == 3 || flag == 1) { //001,011
				originEdges.push_back(TriTuple(x, y, w));
			}
			if (flag == 3 || flag == 2) { //010,011
				originEdges.push_back(TriTuple(y, x, w));
			}
			progress.next(lines + 1);
		}
		e = originEdges.size();
		bfin.close();
		return originEdges;
	}
	vector<unsigned> CHReader::getOrders()
	{
		ifstream bfin(filename, ios::in | ios::binary);

		vector<unsigned> orders;
		orders.resize(v);
		bfin.seekg(4* 5,ios::beg); // skip header
		for (int i = 0; i < v; i++) {
			unsigned order;
			bfin.read((char*)&order, sizeof(order));
			orders[i] = order;
		}
		bfin.close();
		return orders;
	}
	vector<ShortCut> CHReader::getAddEdges()
	{
		ifstream bfin(filename, ios::in | ios::binary);
		bfin.seekg(4*3,ios::beg);
		unsigned m1,m2;
		bfin.read((char*)&m1, sizeof(m1));
		bfin.read((char*)&m2, sizeof(m2));
		bfin.seekg((5 + v + m1 * 4) * 4, ios::beg);

		vector<ShortCut> addEdges;
		addEdges.reserve(m2);
		for (int lines = 0; lines < m2; lines++) {
			node_t x, y,r;
			weight_t w;
			int flag;
			bfin.read((char*)&x, sizeof(node_t));
			bfin.read((char*)&y, sizeof(node_t));
			bfin.read((char*)&w, sizeof(node_t));
			bfin.read((char*)&flag, sizeof(node_t));
			bfin.read((char*)&r, sizeof(node_t));
			if (flag != 7 && flag != 6 && flag != 5) {
				__ERROR("flag uncorrect")
			}
			if (flag == 7 || flag == 5) { //101,111
				addEdges.push_back(ShortCut(x, y, w, r));
			}
			if (flag == 7 || flag == 6) { //110,111
				addEdges.push_back(ShortCut(y, x, w, r));
			}
		}
		int mask;
		bfin.read((char*)&mask, sizeof(int));
		bfin.close();
		return addEdges;
	}
}

