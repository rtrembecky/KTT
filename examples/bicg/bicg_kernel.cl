/**
 * bicg.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;

__kernel void bicgKernel1(__global DATA_TYPE *A, __global DATA_TYPE *p, __global DATA_TYPE *q, int nx, int ny) 
{
    	int i = get_global_id(0);
	
	if (i < nx)
	{
		q[i] = 0.0;

		int j;
		for(j=0; j < ny; j++)
		{
			q[i] += A[i * ny + j] * p[j];
		}
	}
	
}

__kernel void bicgKernel2(__global DATA_TYPE *A, __global DATA_TYPE *r, __global DATA_TYPE *s, int nx, int ny) 
{
	int j = get_global_id(0);
	
	if (j < ny)
	{
		s[j] = 0.0;

		int i;
		for(i = 0; i < nx; i++)
		{
			s[j] += A[i * ny + j] * r[i];
		}
	}
	
}

__kernel void bicgFused(__global DATA_TYPE *A, __global DATA_TYPE *p, __global DATA_TYPE *q, __global DATA_TYPE *r, __global DATA_TYPE *s, int nx, int ny)
{
	/*int j = get_global_id(0);
	int i = get_global_id(1);
	int y = get_local_id(0);
	int x = get_local_id(1);

	float Pvalue = 0;
	// each thread computes one element of the block sub-matrix
	for (int k = 0; k < BLOCK; ++k)
		Pvalue += Md[Row*BLOCK + k] * Nd[k*BLOCK + Col];
	Pd[Row*Width + Col] = Pvalue;
	*/
	
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int bx = get_group_id(0);
	int by = get_group_id(1);
	int gy = get_global_size(1);

	__local float s_A[32][33];
	__local float s_x1[32];
	__local float s_x2[32];

	float l_sum = 0.0f;

	// load x2
	if (ty == 0)
		s_x2[tx] = x2[bx * 32 + tx];

#pragma unroll 1
	for (int i = m*by; i < m*(by + 1); i += 32) {
		// load x1
		if (ty == 1)
			s_x1[tx] = x1[i + tx];
		barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
		for (int j = 0; j < 32; j += BICG_STEP) {
			s_A[ty + j][tx] = A[(i + ty + j)*n + bx * 32 + tx];
			l_sum += s_A[ty + j][tx] * s_x1[ty + j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		float tmp = 0.0f;
#pragma unroll
		for (int j = 0; j < 32; j += BICG_STEP)
			tmp += s_A[tx][ty + j] * s_x2[ty + j];
		s_A[tx][ty] = tmp;
		barrier(CLK_LOCAL_MEM_FENCE);
#if BICG_BATCH <= 1
		if (ty < 16)
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 16];
		barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if BICG_BATCH <= 2
		if (ty < 8)
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 8];
		barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if BICG_BATCH <= 4
		if (ty < 4)
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 4];
		barrier(CLK_LOCAL_MEM_FENCE);
#endif
#if BICG_BATCH <= 8
		if (ty < 2)
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 2];
		barrier(CLK_LOCAL_MEM_FENCE);
#endif
		if (ty == 0) {
#if OPTIMIZE == 2
			atomicAdd(y2 + i + tx, tmp + s_A[tx][1]);
#else
			y2[i + tx + bx*m*gy] = tmp + s_A[tx][1];
#endif
		}

	}

	// compute total sum
	barrier(CLK_LOCAL_MEM_FENCE);
	s_A[ty][tx] = l_sum;
#if BICG_BATCH <= 2
	if (ty < 8) {
		barrier(CLK_LOCAL_MEM_FENCE);
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 8][tx];
#endif
#if BICG_BATCH <= 4
		if (ty < 4) {
			barrier(CLK_LOCAL_MEM_FENCE);
			s_A[ty][tx] = l_sum = l_sum + s_A[ty + 4][tx];
#endif
#if BICG_BATCH <= 8
			if (ty < 2) {
				barrier(CLK_LOCAL_MEM_FENCE);
				s_A[ty][tx] = l_sum = l_sum + s_A[ty + 2][tx];
#endif
				if (ty == 0) {
					barrier(CLK_LOCAL_MEM_FENCE);
					atomicAdd(y1 + bx * 32 + tx, l_sum + s_A[1][tx]);
				}
#if BICG_BATCH <= 8
			}
#endif
#if BICG_BATCH <= 4
		}
#endif
#if BICG_BATCH <= 2
	}
#endif
}

}

// Setup the execution configuration
size_t cl_DimBlock[2], cl_DimGrid[2];
cl_DimBlock[0] = TILE_WIDTH;
cl_DimBlock[1] = TILE_WIDTH;
cl_DimGrid[0] = Width;
cl_DimGrid[1] = Width;
clSetKernelArg(clkern, 0, sizeof(cl_mem), (void*)(&deviceP));
clSetKernelArg(clkern, 1, sizeof(cl_mem), (void*)(&deviceM));
clSetKernelArg(clkern, 2, sizeof(cl_mem), (void*)(&deviceN));
clSetKernelArg(clkern, 3, sizeof(int), (void *)(&Width));
// Launch the device kernel
clEnqueueNDRangeKernel(clcmdque, clkern, 2, NULL,
	cl_DimGrid, cl_DimBlock, 0, NULL,
	&DeviceDone);