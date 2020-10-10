#pragma once

#if __linux__
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#endif

#include <iostream>
#include <locale>			// numpunct<char>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>      	// setprecision
#include <cstdlib>			// exit
#include <ostream>			// color
#include <algorithm>		// sorting
#include <exception>		// sorting
#include <unordered_map>
namespace StreamModifier {
	/**
	 * @enum Color change the color of the output stream
	 */
	enum Color {
		/** <table border=0><tr><td><div> Red </div></td><td><div style="background:#FF0000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_RED = 31, /** <table border=0><tr><td><div> Green </div></td><td><div style="background:#008000;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_GREEN = 32, /** <table border=0><tr><td><div> Yellow </div></td><td><div style="background:#FFFF00;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_YELLOW = 33, /** <table border=0><tr><td><div> Blue </div></td><td><div style="background:#0000FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_BLUE = 34, /** <table border=0><tr><td><div> Magenta </div></td><td><div style="background:#FF00FF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_MAGENTA = 35, /** <table border=0><tr><td><div> Cyan </div></td><td><div style="background:#00FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_CYAN = 36, /** <table border=0><tr><td><div> Light Gray </div></td><td><div style="background:#D3D3D3;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_GRAY = 37, /** <table border=0><tr><td><div> Dark Gray </div></td><td><div style="background:#A9A9A9;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_D_GREY = 90, /** <table border=0><tr><td><div> Light Red </div></td><td><div style="background:#DC143C;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_RED = 91, /** <table border=0><tr><td><div> Light Green </div></td><td><div style="background:#90EE90;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_GREEN = 92, /** <table border=0><tr><td><div> Light Yellow </div></td><td><div style="background:#FFFFE0;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_YELLOW = 93, /** <table border=0><tr><td><div> Light Blue </div></td><td><div style="background:#ADD8E6;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_BLUE = 94, /** <table border=0><tr><td><div> Light Magenta </div></td><td><div style="background:#EE82EE;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_MAGENTA = 95, /** <table border=0><tr><td><div> Light Cyan </div></td><td><div style="background:#E0FFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_L_CYAN = 96, /** <table border=0><tr><td><div> White </div></td><td><div style="background:#FFFFFF;width:20px;height:20px;border:1px solid #000"></div></td></tr></table> */
		FG_WHITE = 97, /** Default */
		FG_DEFAULT = 39
	};

	/**
	 * @enum Emph
	 */
	enum Emph {
		SET_BOLD = 1,
		SET_DIM = 2,
		SET_UNDERLINE = 4,
		SET_RESET = 0,
	};

	/// @cond
	std::ostream& operator<<(std::ostream& os, const Color& mod);
	std::ostream& operator<<(std::ostream& os, const Emph& mod);
	//struct myseps;
	/// @endcond

	void thousandSep();
	void resetSep();
	void fixedFloat();
	void scientificFloat();
}

namespace fUtil {

	template<bool PRINT>
	void print_info(const char* msg) {
		if (PRINT)
			std::cout << msg << std::endl;
	}

	template<typename T>
	std::string typeStringObj(T Obj);
	template<typename T>
	std::string typeString();

	void memInfoPrint(size_t total, size_t free, size_t Req);
	void memInfoHost(int Req);

	template<bool FAULT = true, typename T, typename R>
	bool Compare(T* ArrayA, R* ArrayB, const int size);

	template<bool FAULT = true, typename T, typename R>
	bool Compare(T* ArrayA, R* ArrayB, const int size, bool(*equalFunction)(T, R));

	template<bool FAULT = true, typename T, typename R>
	bool CompareAndSort(T* ArrayA, R* ArrayB, const int size);

	bool isDigit(std::string str);

	class Progress {
	private:
		long long int progressC, nextChunk, total;
		double fchunk;
	public:
		Progress(long long int total);
		~Progress();
		void next(long long int progress);
		void perCent(long long int progress);
	};

	template<typename T, typename R>
	class UniqueMap : public std::unordered_map<T, R> {
	public:
		R insertValue(T id);
	};
}

namespace fileUtil {

	void checkRegularFile(const char* File);
	void checkRegularFile(std::ifstream& fin);
	std::string extractFileName(std::string s);
	std::string extractFileExtension(std::string str);
	std::string extractFilePath(std::string str);

	long long int fileSize(const char* File);
	void skipLines(std::istream& fin, const int nof_lines = 1);
}
