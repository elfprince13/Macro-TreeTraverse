//
//  cudahelper.h
//  TreeTraverse
//
//  Created by Thomas Dickerson on 3/15/15.
//  Copyright (c) 2015 StickFigure Graphic Productions. All rights reserved.
//

#ifndef TreeTraverse_cudahelper_h
#define TreeTraverse_cudahelper_h

#include <cuda.h>
#include <stdio.h>


inline const char* cuPrintError(int err)
{
	switch (err) {
		case CUDA_SUCCESS : return "CUDA_SUCCESS";
		case CUDA_ERROR_INVALID_VALUE : return "CUDA_ERROR_INVALID_VALUE";
		case CUDA_ERROR_OUT_OF_MEMORY : return "CUDA_ERROR_OUT_OF_MEMORY";
		case CUDA_ERROR_NOT_INITIALIZED : return "CUDA_ERROR_NOT_INITIALIZED";
		case CUDA_ERROR_DEINITIALIZED : return "CUDA_ERROR_DEINITIALIZED";
		case CUDA_ERROR_NO_DEVICE : return "CUDA_ERROR_NO_DEVICE";
		case CUDA_ERROR_INVALID_DEVICE : return "CUDA_ERROR_INVALID_DEVICE";
		case CUDA_ERROR_INVALID_IMAGE : return "CUDA_ERROR_INVALID_IMAGE";
		case CUDA_ERROR_INVALID_CONTEXT : return "CUDA_ERROR_INVALID_CONTEXT";
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT : return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
		case CUDA_ERROR_MAP_FAILED : return "CUDA_ERROR_MAP_FAILED";
		case CUDA_ERROR_UNMAP_FAILED : return "CUDA_ERROR_UNMAP_FAILED";
		case CUDA_ERROR_ARRAY_IS_MAPPED : return "CUDA_ERROR_ARRAY_IS_MAPPED";
		case CUDA_ERROR_ALREADY_MAPPED : return "CUDA_ERROR_ALREADY_MAPPED";
		case CUDA_ERROR_NO_BINARY_FOR_GPU : return "CUDA_ERROR_NO_BINARY_FOR_GPU";
		case CUDA_ERROR_ALREADY_ACQUIRED : return "CUDA_ERROR_ALREADY_ACQUIRED";
		case CUDA_ERROR_NOT_MAPPED : return "CUDA_ERROR_NOT_MAPPED";
		case CUDA_ERROR_INVALID_SOURCE : return "CUDA_ERROR_INVALID SOURCE";
		case CUDA_ERROR_FILE_NOT_FOUND : return "CUDA_ERROR_FILE_NOT_FOUND";
		case CUDA_ERROR_INVALID_HANDLE : return "CASE_ERROR_INVALID_HANDLE";
		case CUDA_ERROR_NOT_FOUND : return "CUDA_ERROR_NOT_FOUND";
		case CUDA_ERROR_NOT_READY : return "CUDA_ERROR_NOT_READY";
		case CUDA_ERROR_LAUNCH_FAILED : return "CUDA_ERROR_LAUNCH_FAILED";
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : return "CUDA_ERROR_LAUNCH_OUT_OF_RESOUCES";
		case CUDA_ERROR_LAUNCH_TIMEOUT : return "CUDA_ERROR_LAUNCH_TIMEOUT";
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING : return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
		case CUDA_ERROR_UNKNOWN : return "CUDA_ERROR_UNKNOWN";
		default :
			char buff[256];
			sprintf(buff,"%s : %d .", "Unknown CUresult", err);
			fprintf(stderr,"%s\n", buff);
			return "Unknown CUresult.";
			//return buff;
	}
}


#  define CU_SAFE_CALL_NO_SYNC( call ) {                                     \
CUresult err = call;                                                     \
if( CUDA_SUCCESS != err) {                                               \
fprintf(stderr, "Cuda driver error <%s> in file '%s' in line %i.\n",   \
cuPrintError(err), __FILE__, __LINE__ );                                   \
exit(EXIT_FAILURE);                                                  \
} }

#  define CU_SAFE_CALL( call )       CU_SAFE_CALL_NO_SYNC(call);

#  define CU_SAFE_CALL_NO_SYNC_KERNEL( call , kernel ) {                                     \
CUresult err = call;                                                     \
if( CUDA_SUCCESS != err) {                                               \
fprintf(stderr, "Cuda driver error <%s> in file '%s' in line %i. (Kernel: %s)\n",   \
cuPrintError(err), __FILE__, __LINE__, kernel );                                   \
exit(EXIT_FAILURE);                                                  \
} }

#  define CU_SAFE_CALL_KERNEL( call , kernel )       CU_SAFE_CALL_NO_SYNC_KERNEL(call, kernel);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#endif
