#include <boost/property_tree/ptree.hpp>  
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem/fstream.hpp>

#include <stdio.h>
#include <chrono>
#include <algorithm>
#include "graph.h"
#include "cudaGraph.cuh"
#include "tensor.h"
#include "groupReadKernel.cuh"
using namespace graph;
using namespace cuda_graph;

#define make_sure(expression,msg)\
do{\
if(!(expression))\
{\
	__ERROR(msg) \
}\
}while(0)


int main(int argc, char* argv[])
{
	make_sure(argc==3,"specify n");
	int nn = atoi(argv[1]);
	int ggg = atoi(argv[2]);
    const int blocks = 272 * 8;
    const int threads = 256;
	for(int n = nn;n<=272 * 8 * 256 * 4;n+=nn){
		int temp;
		Tensor<int> in = Tensor<int>::randint(n);
		Tensor<int> out = Tensor<int>::empty(n);
		// index.print(32);
		in.copyToDevice();
		out.copyToDevice();
		cudaError_t err = cudaGetLastError();
		if (cudaSuccess != err) {
			cout<<"error copy"<<endl;
			std::exit(EXIT_FAILURE);
		}
		auto time1 = chrono::high_resolution_clock::now();
		for(int gg=0;gg<ggg;gg++){
			hbftest_copy_cuda_kernel<int><<<blocks, threads>>>(
				in.device,out.device,n
			);
		}
		cudaDeviceSynchronize();
		auto time2 = chrono::high_resolution_clock::now();
		err = cudaGetLastError();
		if (cudaSuccess != err) {
			cout<<"error kernel"<<endl;
			std::exit(EXIT_FAILURE);
		}
		auto tt1 = chrono::duration_cast<chrono::microseconds>(time2 - time1).count() * 0.001;
		cout << n << " " << tt1 << endl;
	}	
    return 0;
}