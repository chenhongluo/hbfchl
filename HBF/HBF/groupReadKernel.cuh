template <typename T>
__global__ void hbftest_groupread_cuda_kernel(
    T* input,
    T* output,
	int* index,
	int nn,
	int kk
    )
{
	const int realID =  blockIdx.x * blockDim.x +  threadIdx.x;
	const int IDStride = gridDim.x * blockDim.x;

    T sum = 0.0;
        for(int i = realID;i<nn;i+=IDStride)
        {
			int j = index[i];
			for(int k = 0;k<kk;k++){
				__syncwarp(0xFFFFFFFF);
            	sum += input[k * nn + j];
        	}
		}
	output[realID] = sum;
}

template <typename T>
__global__ void hbftest_copy_cuda_kernel(
    T* input,
    T* output,
	int nn
    )
{
	const int realID =  blockIdx.x * blockDim.x +  threadIdx.x;
	const int IDStride = gridDim.x * blockDim.x;

        for(int i = realID;i<nn;i+=IDStride)
        {
			output[i] = input[i];
		}
}