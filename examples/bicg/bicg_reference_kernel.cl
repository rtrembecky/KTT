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

// process BICG_BATCH elements in thread
#define BICG_BATCH 8
#define BICG_STEP 32/BICG_BATCH

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

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
	union{
		unsigned int u32;
		float f32;
	} next, expected, current;
	current.f32 = *addr;
	do{
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr,
		expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}

__kernel void bicgFusedRef(__global DATA_TYPE *A, __global DATA_TYPE *x1, __global DATA_TYPE *y1, __global DATA_TYPE *x2, __global DATA_TYPE *y2, int m, int n)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int bx = get_group_id(0);
	int by = get_group_id(1);
	int gy = get_global_size(1);

	__local DATA_TYPE s_A[32][33];
	__local DATA_TYPE s_x1[32];
	__local DATA_TYPE s_x2[32];

	float l_sum = 0.0f;
	//if (ty < 1 && tx < 1 && by == 0)
	//	printf("tx,ty: %d,%d, bx,by: %d,%d, gy: %d\n m: %d, n: %d, x1[0]: %2.2f, y1[0]: %2.2f\n", tx, ty, bx, by, gy, m, n, x1[0], y1[0]);

	// load x2
	if (ty == 0)
		s_x2[tx] = x2[bx * 32 + tx];
	for (int i = m*by; i < m*(by + 1); i += 32) {
		// load x1
		if (ty == 1)
			s_x1[tx] = x1[i + tx];
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int j = 0; j < 32; j += BICG_STEP) {
			s_A[ty + j][tx] = A[(i + ty + j)*n + bx * 32 + tx];
			l_sum += s_A[ty + j][tx] * s_x1[ty + j];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		float tmp = 0.0f;

		for (int j = 0; j < 32; j += BICG_STEP)
			tmp += s_A[tx][ty + j] * s_x2[ty + j];
		s_A[tx][ty] = tmp;
		barrier(CLK_LOCAL_MEM_FENCE);
//BATCH 8 code
		if (ty < 2)
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 2];
		barrier(CLK_LOCAL_MEM_FENCE);

		if (ty == 0) {
//#if OPTIMIZE == 2 (atomics, no need to reduce later)
			atomicAdd_g_f(y2 + i + tx, tmp + s_A[tx][1]);
//#else
			//y2[i + tx + bx*m*gy] = tmp + s_A[tx][1];
		}
	}

	// compute total sum
	barrier(CLK_LOCAL_MEM_FENCE);
	s_A[ty][tx] = l_sum;
//BATCH 8 code
	if (ty < 2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 2][tx];
//end
		if (ty == 0) {
			barrier(CLK_LOCAL_MEM_FENCE);
			atomicAdd_g_f(y1 + bx * 32 + tx, l_sum + s_A[1][tx]);
		}
	}
}
