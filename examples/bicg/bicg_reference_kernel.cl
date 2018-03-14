__kernel void directCoulombSumReference(__global float4* atomInfo, int numberOfAtoms, float gridSpacing, __global float* energyGrid)
{
    int xIndex = get_global_id(0);
    int yIndex = get_global_id(1);
        
    int outIndex = get_global_size(0) * yIndex + xIndex;

    float currentEnergy = energyGrid[outIndex];

    float coordX = gridSpacing * xIndex;
    float coordY = gridSpacing * yIndex;
    float energyValue = 0.0f;

    for (int i = 0; i < numberOfAtoms; i++)
    {
        float distanceX = coordX - atomInfo[i].x;
        float distanceY = coordY - atomInfo[i].y;
        float partialResult = native_rsqrt(distanceX * distanceX + distanceY * distanceY + atomInfo[i].z);
        energyValue += atomInfo[i].w * partialResult;
    }

    energyGrid[outIndex] += currentEnergy + energyValue;
}

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
