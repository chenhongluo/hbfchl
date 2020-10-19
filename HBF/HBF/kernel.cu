
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "graph.h"
using namespace std;
using namespace graph;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void test() 
{
	vector<TriTuple> ts;
	ts.push_back(TriTuple(0, 2, 5));
	ts.push_back(TriTuple(1, 2, 1));
	ts.push_back(TriTuple(1, 3, 1));
	ts.push_back(TriTuple(1, 4, 1));
	ts.push_back(TriTuple(2, 4, 2));
	ts.push_back(TriTuple(2, 3, 2));
	ts.push_back(TriTuple(3, 4, 3));
	ts.push_back(TriTuple(4, 1, 4));
	GraphWeight hostGraph(5, 8, EdgeType::UNDIRECTED, ts);
	//EdgeType userEdgeType = EdgeType::UNDEF_EDGE_TYPE;
	//IntRandomUniform ir = IntRandomUniform();
	//GraphRead* reader = getGraphReader("F:/data_graph/delaunay_n20.graph", userEdgeType,ir);
	//GraphWeight hostGraph = GraphWeight(reader);
	//vector<int2> v1 = hostGraph.getOutEdgesOfNode(0);
	//vector<int2> v2 = hostGraph.getInEdgesOfNode(186869);
	hostGraph.toCSR();
	GraphHost hg(hostGraph);
	vector<dist_t> res1, res2;
	double t1, t2;
	hg.computeAndTick(0, res1, t1, GraphHost::HostSelector::BoostD);
	hg.computeAndTick(0, res2, t2, GraphHost::HostSelector::BellmanFord);
}

int main()
{
	// 读取参数配置data和*.config
	// config :cuda config ,graph config, nodeSize
	// 构建图
	// 生成GraphHost和CudaGraph
	// 完成后生成source nodes开始search
	// 取得结果后对比
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
