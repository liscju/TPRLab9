#include <iostream>
#include <cublas.h>

void fun()
{
	const int vector_size = 200;
	double *CPU_A, *CPU_B, *CPU_C;
	double *GPU_A, *GPU_B, *GPU_C;

	CPU_A = new double[vector_size];
	CPU_B = new double[vector_size];
	CPU_C = new double[vector_size];

	for(int i = 0; i < vector_size; i++)
	{
		CPU_A[i] = static_cast<double>(i);
		CPU_B[i] = static_cast<double>(i);
	}
	
	cublasAlloc(vector_size, sizeof(double), (void **) &GPU_A);
	cublasAlloc(vector_size, sizeof(double), (void **) &GPU_B);
	cublasAlloc(vector_size, sizeof(double), (void **) &GPU_C);
	
	cublasSetVector(vector_size, sizeof(double), CPU_A, 1, GPU_A, 1);
	cublasSetVector(vector_size, sizeof(double), CPU_B, 1, GPU_B, 1);

	cublasDaxpy(vector_size, 1.0, GPU_A, 1, GPU_B, 1);

	cublasGetVector(vector_size, sizeof(double), GPU_B, 1, CPU_C, 1);

	for(int i = 0; i < vector_size; i++)
	{
		std::cout << CPU_C[i] << std::endl;
	}

	delete[] CPU_A;
	delete[] CPU_B;
	delete[] CPU_C;
	
	cublasFree(GPU_A);
	cublasFree(GPU_B);
	cublasFree(GPU_C);
}

int main()
{
	cublasInit();
	int version;
	cublasGetVersion(&version);
	std::cout << "Cublas version:" << version << std::endl;
	fun();
	cublasShutdown();
	return 0;
}

