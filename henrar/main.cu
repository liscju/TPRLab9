#include <iostream>
#include <cublas.h>

int main()
{
	cublasInit();
	int version;
	cublasGetVersion(&version);
	std::cout << "Cublas version:" << version << std::endl;
	cublasShutdown();
	return 0;
}

