#include <iostream>
#include <cstdlib>
#include <cublas.h>



int main(int argc, char** argv) {
	if(argc != 2) {
		fprintf(stderr, "Wrong arguments. Usage: %s <N>\n", argv[0]);
		return EXIT_FAILURE;
	}
	int N = atoi(argv[1]);

	cublasInit();
	int version;
	cublasGetVersion(&version);
	std::cout << "Cublas version:" << version << std::endl;

	double *CPU_A, *CPU_B, *CPU_C;
	double *GPU_A, *GPU_B, *GPU_C;
	
	CPU_A = new double[N];
	CPU_B = new double[N];
	CPU_C = new double[N];

	for(int i=0; i<N; i++) {
		CPU_A[i] = (double)i;
		CPU_B[i] = (double)i;
	}

	cublasAlloc(N, sizeof(double), (void **) &GPU_A);
	cublasAlloc(N, sizeof(double), (void **) &GPU_B);
	cublasAlloc(N, sizeof(double), (void **) &GPU_C);

	//cublasDaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy);

	cublasSetVector(N, sizeof(double), CPU_A, 1, GPU_A, 1);
	cublasSetVector(N, sizeof(double), CPU_B, 1, GPU_B, 1);

	cublasDaxpy(N, GPU_A, GPU_B, 1, GPU_C, 1);

	cublasGetVector(N, sizeof(double), GPU_C, 1, CPU_C, 1);

	for(int i=0; i<N; i++) {
		std::cout << CPU_C[i] << " ";
	}
	std::cout << std::endl;

	cublasFree(GPU_A);
	cublasFree(GPU_B);
	cublasFree(GPU_C);

	delete[] CPU_A;
	delete[] CPU_B;
	delete[] CPU_C;

	cublasShutdown();
	return 0;
}
