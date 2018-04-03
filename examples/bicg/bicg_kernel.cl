/**
 * bicg.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

// process BICG_BATCH elements in thread
#define BICG_STEP 32/BICG_BATCH

typedef float DATA_TYPE;

__kernel void bicgKernel1(__global DATA_TYPE *A, __global DATA_TYPE *p, __global DATA_TYPE *q, int nx, int ny)
{
	int i = get_global_id(0);

	if (i < nx)
	{
		q[i] = 0.0;

		int j;
		for (j = 0; j < ny; j++)
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
		for (i = 0; i < nx; i++)
		{
			s[j] += A[i * ny + j] * r[i];
		}
	}
}

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
	union {
		unsigned int u32;
		float f32;
	} next, expected, current;
	current.f32 = *addr;
	do {
		expected.f32 = current.f32;
		next.f32 = expected.f32 + val;
		current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
			expected.u32, next.u32);
	} while (current.u32 != expected.u32);
}

inline void barrier_sh_red() {
#if USE_SHARED_REDUCTION == 1
	barrier(CLK_LOCAL_MEM_FENCE);
#else
	barrier(CLK_GLOBAL_MEM_FENCE);
#endif
}

__kernel void bicgFused(__global DATA_TYPE *A, __global DATA_TYPE *x1, __global DATA_TYPE *y1, __global DATA_TYPE *x2, __global DATA_TYPE *y2, int m, int n)
{
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	int bx = get_group_id(0);
	int by = get_group_id(1);
	int gy = get_global_size(1);
	int wg_y_num = get_num_groups(1);
	//m = m*wg_y_num;

//#if USE_SHARED_MATRIX == 1
	__local DATA_TYPE s_A[32][33];
//#endif
#if USE_SHARED_VECTOR1 == 1
	__local DATA_TYPE s_x1[32];
#endif
#if USE_SHARED_VECTOR2 == 1
	__local DATA_TYPE s_x2[32];
#endif

	float l_sum = 0.0f;

	// load x2
#if USE_SHARED_VECTOR2 == 1
	if (ty == 0) // first row of workgroup
		//s_x2[tx] = x2[bx * 32 + tx]; // load x2 at [0-127]*32+[0-31] = [0-4064]+[0-31]
		s_x1[tx] = x1[bx * 32 + tx];
#endif

#pragma unroll 1
	//for (int i = m*by; i < m*(by + 1); i += 32) { //i = m*by = 256*[0-15] = [0-3840]; i <= [224-4064];  8 iterations,  by = 0: i=0,32,64,96,128,160,192,224
	for (int i = (n)*by; i < (n)*(by + 1); i += 32) {

		// load x1
#if USE_SHARED_VECTOR1 == 1
		if (ty == 1) // second row of workgroup
			//s_x1[tx] = x1[i + tx]; // 16 y-workgroups times 256 (m) parts, which is divided to 8 i-iterations with 32 x-threads
			s_x2[tx] = x2[i + tx];
		barrier(CLK_LOCAL_MEM_FENCE);
#endif

// multiply x1
#pragma unroll
		for (int j = 0; j < 32; j += BICG_STEP) { // STEP is 4 for BATCH 8, means 8 iterations for j: 0,4,8,12,16,20,24,28
#if USE_SHARED_MATRIX == 1
			//s_A[ty + j][tx] = A[(i + ty + j)*n + bx * 32 + tx]; // A at ([0-4064]+[0-3]+[0-28])*4096 + [0-127]*32 + [0-31], loaded to s_A at row [0-3]+[0-28] and column [0-31]
			s_A[ty + j][tx] = A[(i + ty + j)*m + bx * 32 + tx];
	#if USE_SHARED_VECTOR1 == 1
			//l_sum += s_A[ty + j][tx] * s_x1[ty + j];
			l_sum += s_A[ty + j][tx] * s_x2[ty + j];
	#else
			l_sum += s_A[ty + j][tx] * x1[i + ty + j];
			if (bx == 0 && by == 0 && tx == 0 && ty == 0)
				;// printf("[0,0] l_sum: %f\n  x1[i + ty + j]: %f\n  y1: %f %f %f %f\n", l_sum, x1[i + ty + j], y1[0], y1[1], y1[2], y1[3]);
	#endif // USE_SHARED_VECTOR1
#else
	#if USE_SHARED_VECTOR1 == 1
			l_sum += A[(i + ty + j)*n + bx * 32 + tx] * s_x1[ty + j];
	#else
			l_sum += A[(i + ty + j)*n + bx * 32 + tx] * x1[i + ty + j];
	#endif // USE_SHARED_VECTOR1
#endif // USE_SHARED_MATRIX
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		float tmp = 0.0f;

// multiply x2
#pragma unroll
		for (int j = 0; j < 32; j += BICG_STEP)
#if USE_SHARED_MATRIX == 1
	#if USE_SHARED_VECTOR2 == 1
			//tmp += s_A[tx][ty + j] * s_x2[ty + j];
			tmp += s_A[tx][ty + j] * s_x1[ty + j];
	#else
			tmp += s_A[tx][ty + j] * x2[i + ty + j];
	#endif // USE_SHARED_VECTOR2
		s_A[tx][ty] = tmp;
		barrier(CLK_LOCAL_MEM_FENCE);
#else
	#if USE_SHARED_VECTOR2 == 1
			tmp += A[(i + tx)*n + bx * 32 + ty + j] * s_x2[ty + j];
	#else
			tmp += A[(i + tx)*n + bx * 32 + ty + j] * x2[i + ty + j];
	#endif
		A[(i + tx)*n + bx * 32 + ty] = tmp;
		barrier(CLK_GLOBAL_MEM_FENCE);
#endif // USE_SHARED_MATRIX

#if BICG_BATCH <= 1
		if (ty < 16)
	#if USE_SHARED_MATRIX == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 16];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*n + bx * 32 + ty] = tmp = tmp + A[(i + tx)*n + bx * 32 + ty + 16];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_MATRIX
#endif // BICG_BATCH

#if BICG_BATCH <= 2
		if (ty < 8)
	#if USE_SHARED_MATRIX == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 8];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*n + bx * 32 + ty] = tmp = tmp + A[(i + tx)*n + bx * 32 + ty + 8];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_MATRIX
#endif // BICG_BATCH

#if BICG_BATCH <= 4
		if (ty < 4)
	#if USE_SHARED_MATRIX == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 4];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*n + bx * 32 + ty] = tmp = tmp + A[(i + tx)*n + bx * 32 + ty + 4];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_MATRIX
#endif // BICG_BATCH

#if BICG_BATCH <= 8
		if (ty < 2)
	#if USE_SHARED_MATRIX == 1
			s_A[tx][ty] = tmp = tmp + s_A[tx][ty + 2];
		barrier(CLK_LOCAL_MEM_FENCE);
	#else
			A[(i + tx)*n + bx * 32 + ty] = tmp = tmp + A[(i + tx)*n + bx * 32 + ty + 2];
		barrier(CLK_GLOBAL_MEM_FENCE);
	#endif // USE_SHARED_MATRIX
#endif // BICG_BATCH

		if (ty == 0)
#if ATOMICS == 1
	#if USE_SHARED_MATRIX == 1
			//atomicAdd_g_f(y2 + i + tx, tmp + s_A[tx][1]);
			atomicAdd_g_f(y1 + i + tx, tmp + s_A[tx][1]);
	#else
			atomicAdd_g_f(y2 + i + tx, tmp + A[(i + tx)*n + bx * 32 + 1]);
	#endif // USE_SHARED_MATRIX
#else // TODO: reduce in manipulator
	#if USE_SHARED_MATRIX == 1
			y2[i + tx + bx*m*gy] = tmp + s_A[tx][1];
	#else
			y2[i + tx + bx*m*gy] = tmp + A[(i + tx)*n + bx * 32 + 1];
	#endif // USE_SHARED_MATRIX
#endif // ATOMICS
	}

	// compute total sum
	barrier(CLK_LOCAL_MEM_FENCE);
#if USE_SHARED_REDUCTION == 1
	s_A[ty][tx] = l_sum;
#else
	A[(m*by + 7*32 + ty)*n + bx * 32 + tx] = l_sum;
#endif // USE_SHARED_REDUCTION

#if BICG_BATCH <= 1
	barrier_sh_red();
	if (ty < 16)
	#if USE_SHARED_REDUCTION == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 16][tx];
	#else
		A[(m*by + 7 * 32 + ty)*n + bx * 32 + tx] = l_sum = l_sum + A[(m*by + 7 * 32 + ty + 16)*n + bx * 32 + tx];
	#endif // USE_SHARED_REDUCTION
#endif // BICG_BATCH

#if BICG_BATCH <= 2
	barrier_sh_red();
	if (ty < 8)
	#if USE_SHARED_REDUCTION == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 8][tx];
	#else
		A[(m*by + 7*32 + ty)*n + bx * 32 + tx] = l_sum = l_sum + A[(m*by + 7*32 + ty + 8)*n + bx * 32 + tx];
	#endif // USE_SHARED_REDUCTION
#endif // BICG_BATCH

#if BICG_BATCH <= 4
	barrier_sh_red();
	if (ty < 4)
	#if USE_SHARED_REDUCTION == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 4][tx];
	#else
		A[(m*by + 7 * 32 + ty)*n + bx * 32 + tx] = l_sum = l_sum + A[(m*by + 7 * 32 + ty + 4)*n + bx * 32 + tx];
	#endif // USE_SHARED_REDUCTION
#endif // BICG_BATCH

#if BICG_BATCH <= 8
	barrier_sh_red();
	if (ty < 2)
	#if USE_SHARED_REDUCTION == 1
		s_A[ty][tx] = l_sum = l_sum + s_A[ty + 2][tx];
	#else
		A[(m*by + 7 * 32 + ty)*n + bx * 32 + tx] = l_sum = l_sum + A[(m*by + 7 * 32 + ty + 2)*n + bx * 32 + tx];
	#endif
#endif // BICG_BATCH

	barrier_sh_red();
	if (ty == 0)
#if ATOMICS == 1
	#if USE_SHARED_REDUCTION == 1
		//atomicAdd_g_f(y1 + bx * 32 + tx, l_sum + s_A[1][tx]);
		atomicAdd_g_f(y2 + bx * 32 + tx, l_sum + s_A[1][tx]);
	#else
		atomicAdd_g_f(y1 + bx * 32 + tx, l_sum + A[(m*by + 7 * 32 + 1)*n + bx * 32 + tx]);
	#endif // USE_SHARED_REDUCTION
#else // TODO: reduce in manipulator
	#if USE_SHARED_REDUCTION == 1
		y1[bx * 32 + tx + bx*m*gy] = l_sum + s_A[1][tx];
	#else
		y1[bx * 32 + tx + bx*m*gy] = l_sum + A[(m*by + 7 * 32 + 1)*n + bx * 32 + tx];
	#endif // USE_SHARED_REDUCTION
#endif // ATOMICS
	/*if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
		printf("y1: %f %f %f %f\n", y1[0], y1[1], y1[2], y1[3]);
		printf("y2: %f %f %f %f\n", y2[0], y2[1], y2[2], y2[3]);
	}*/
		//printf("3. l_sum for tx,ty 0,0: %f\n  A[m*by + 7*32 + 1]: %f\n  y1: %f %f %f %f\n", l_sum, A[m*by + 7 * 32 + 1], y1[0], y1[1], y1[2], y1[3]);
}
