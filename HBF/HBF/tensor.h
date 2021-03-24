#pragma once
#include<map>
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
template<class T>
class Tensor{
public:
	T* host;
	T* device;
	int size;

	Tensor()
	{

	}

	Tensor(const Tensor &obj)
	{
		host = obj.host;
		device = obj.device;
		size = obj.size;
	}

	void print(int k){
		if(k == -1 || k > size){
			k = size;
		}
		for(int i=0;i<k;i++){
			cout<<host[i]<<",";
		}
		cout<<endl;
	}

	~Tensor()
	{
		delete[] host;
		if(device!=NULL){
			cudaFree(device);
			device = NULL;
		}
	}

	static Tensor empty(int size){
		Tensor t;
		t.size = size;
		t.host = new T[size];
		t.device = NULL;
		return t;
	}

	static Tensor randperm(int size){
		Tensor t = Tensor::empty(size);
		int *arr = (int*)malloc(size*sizeof(int)); 
    	int count = 0;
		memset(arr,0,size*sizeof(int));
		srand(time(NULL));
		while(count<size)
		{
			int val = rand()%size;
			if (!arr[val])
			{
				t.host[count] = val;
				arr[val]=1;
				++count;
			}
		}
		free(arr);
		arr = NULL;
		return t;
	}

	static Tensor randint(int size){
		Tensor t = Tensor::empty(size);
		int *arr = (int*)malloc(size*sizeof(int)); 
    	int count = 0;
		while(count<size)
		{
			int val = rand()%size;
			t.host[count++] = val;
		}
		free(arr);
		arr = NULL;
		return t;
	}

	void copyToDevice(){
		if(device==NULL){
			cudaMalloc((void**)&device, size * sizeof(T));
		}
		cudaMemcpy(device, host, size * sizeof(T), cudaMemcpyHostToDevice);
	}

	void copyToHost(){
		cudaMemcpy(host, device, size * sizeof(T), cudaMemcpyDeviceToHost);
	}
};