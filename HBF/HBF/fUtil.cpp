/*------------------------------------------------------------------------------
Copyright © 2015 by Nicola Bombieri

H-BF is provided under the terms of The MIT License (MIT):

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------*/
/**
 * @author Federico Busato
 * Univerity of Verona, Dept. of Computer Science
 * federico.busato@univr.it
 */
#include <fstream>
#include <locale>
#include <string.h>
#include "fUtil.h"

namespace StreamModifier {
	/// @cond

	std::ostream& operator<<(std::ostream& os, const Color& mod) {
		return os << "\033[" << (int)mod << "m";
	}

	std::ostream& operator<<(std::ostream& os, const Emph& mod) {
		return os << "\033[" << (int)mod << "m";
	}

	struct myseps : std::numpunct<char> {
		char do_thousands_sep() const { return ','; }	// use space as separator
		std::string do_grouping() const { return "\3"; }	// digits are grouped by 3 digits each
	};

	void thousandSep() { std::cout.imbue(std::locale(std::locale(), new myseps)); }
	void resetSep() { std::cout.imbue(std::locale()); }

	void fixedFloat() {
		std::cout.setf(std::ios::fixed, std::ios::floatfield);
	}

	void scientificFloat() {
		std::cout.setf(std::ios::scientific, std::ios::floatfield);
	}
	/// @endcond
} //@StreamModifier

namespace fUtil {

	void memInfoPrint(size_t total, size_t free, size_t Req) {
		std::cout << "  Total Memory:\t" << (total >> 20) << " MB" << std::endl
			<< "   Free Memory:\t" << (free >> 20) << " MB" << std::endl
			<< "Request memory:\t" << (Req >> 20) << " MB" << std::endl
			<< "   Request (%):\t" << ((Req >> 20) * 100) / (total >> 20) << " %" << std::endl << std::endl;
		if (Req > free)
			throw std::runtime_error(" ! Memory too low");
	}

#if __linux__

	void memInfoHost(int Req) {
		long pages = ::sysconf(_SC_PHYS_PAGES);
		long page_size = ::sysconf(_SC_PAGE_SIZE);
		memInfoPrint(pages * page_size, pages * page_size - 100 * (1 << 20), Req);
	}
#endif
	// --------------------------- FILE  ---------------------------------------------------

	bool isDigit(std::string str) {
		return str.find_first_not_of("0123456789") == std::string::npos;
	}

	Progress::Progress(long long int _total) {
		total = _total;
		progressC = 1L;
		fchunk = (double)total / 100;
		nextChunk = fchunk;
		// std::cout << "     0%" << std::flush;
	}

	Progress::~Progress() {
		// std::cout << "\b\b\b\b\b" << " 100% Complete!" << std::endl << std::endl << std::flush;
	}

	void Progress::next(long long int progress) {
		if (progress == nextChunk) {
			// std::cout << "\b\b\b\b\b" << std::setw(4) << progressC++ << "%" << std::flush;
			nextChunk = progressC * fchunk;
		}
	}

	void Progress::perCent(long long int progress) {
		// std::cout << "\b\b\b\b\b" << std::left << std::setw(4) << std::setprecision(1) << ((float)progress / total) * 100 << "%" << std::flush;
	}
}

namespace fileUtil {

	void skipLines(std::istream& fin, const int nof_lines) {
		for (int i = 0; i < nof_lines; ++i)
			fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	long long int fileSize(const char* File) {
		std::ifstream fin(File);
		fin.seekg(0L, std::ios::beg);
		long long int startPos = fin.tellg();
		fin.seekg(0L, std::ios::end);
		long long int endPos = fin.tellg();
		fin.close();
		return endPos - startPos;
	}

	void checkRegularFile(std::ifstream& fin) {
		if (!fin.is_open() || fin.fail() || fin.bad() || fin.eof())
			__ERROR(" Error. Read file: ")

			try {
			char c;	fin >> c;
		}
		catch (std::ios_base::failure& e) {
			__ERROR(" Error. Read file ")
		}
		fin.seekg(0, std::ios::beg);
	}

	void checkRegularFile(const char* File) {
		std::ifstream fin(File);
		if (!fin.is_open() || fin.fail() || fin.bad() || fin.eof())
			__ERROR(" Error. Read file: " << File)

			try {
			char c;	fin >> c;
		}
		catch (std::ios_base::failure& e) {
			__ERROR(" Error. Read file: " << File)
		}
		fin.close();
	}

	std::string extractFilePath(std::string str) {
		return str.substr(0, str.find_last_of("/") + 1);
	}

	std::string extractFileName(std::string str) {
		std::string name2 = str.substr(0, str.find_last_of("."));

		const int found = name2.find_last_of("/");
		if (found >= 0)
			return name2.substr(found + 1);
		return name2;
	}

	std::string extractFileExtension(std::string str) {
		return str.substr(str.find_last_of("."));
	}

} //@fUtil


namespace stringUtil {

	std::vector<std::string> split(const std::string& str, const std::string& delim) {
		std::vector<std::string> res;
		if ("" == str) return res;
		//先将要切割的字符串从string类型转换为char*类型  
		char * strs = new char[str.length() + 1]; //不要忘了  
		strcpy(strs, str.c_str());

		char * d = new char[delim.length() + 1];
		strcpy(d, delim.c_str());

		char *p = strtok(strs, d);
		while (p) {
			std::string s = p; //分割得到的字符串转换为string类型  
			res.push_back(s); //存入结果数组  
			p = strtok(NULL, d);
		}
		return res;
	}
}
