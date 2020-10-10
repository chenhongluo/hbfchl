#pragma once
#include<map>
#include <iostream>
#include <fstream>
#include"graph.h"
using namespace std;
namespace graph {
	typedef vector<TriTuple>&&(*Translator)(ifstream& f);
	class graphRead {
	public:
		map<string, Translator> translators;
	private:
		vector<TriTuple>&& readMatrixMarket(ifstream& f);
		vector<TriTuple>&& readDimacs9(ifstream& f);
		vector<TriTuple>&& readDimacs10(ifstream& f);
		vector<TriTuple>&& readSnap(ifstream& f);
		vector<TriTuple>&& readBinary(ifstream& f);
	};
}