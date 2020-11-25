#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cstdio>
using namespace cooperative_groups;

namespace KernelV4 {
	template <int VW_SIZE>
	__global__ void HBFSearchV4Atomic64(
		int *__restrict__ devNodes,
		int2 *__restrict__ devEdges,
		int2 *__restrict__ devDistances,
		int* devF1, int* devF2,
		int *__restrict__ devSizes,
		const int sharedLimit,
		const int tileLimit,
		int level)
	{

	}
}