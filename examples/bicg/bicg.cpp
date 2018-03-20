#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "tuner_api.h"

#if defined(_MSC_VER)
    #define KTT_KERNEL_FILE "../examples/bicg/bicg_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../examples/bicg/bicg_reference_kernel.cl"
#else
    #define KTT_KERNEL_FILE "../../examples/bicg/bicg_kernel.cl"
    #define KTT_REFERENCE_KERNEL_FILE "../../examples/bicg/bicg_reference_kernel.cl"
#endif

//#include <stdlib.h>
//#include <time.h>
//#include <sys/time.h>
//#include <math.h>
#define MAX_SOURCE_SIZE (0x100000)

/* Problem size. */
#define NX 4096
#define NY 4096

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

typedef float DATA_TYPE;

int main(int argc, char** argv)
{
    // Initialize platform index, device index and paths to kernels
    ktt::PlatformIndex platformIndex = 0;
    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = KTT_KERNEL_FILE;
    std::string referenceKernelFile = KTT_REFERENCE_KERNEL_FILE;

    if (argc >= 2)
    {
        platformIndex = std::stoul(std::string(argv[1]));
        if (argc >= 3)
        {
            deviceIndex = std::stoul(std::string(argv[2]));
            if (argc >= 4)
            {
                kernelFile = std::string(argv[3]);
                if (argc >= 5)
                {
                    referenceKernelFile = std::string(argv[4]);
                }
            }
        }
    }

    // Declare kernel parameters
    const ktt::DimensionVector ndRangeDimensions(256, 256);
    const ktt::DimensionVector workGroupDimensions;
    const ktt::DimensionVector referenceWorkGroupDimensions(16, 16);

    // Declare data variables
    //float gridSpacing = 0.5f;
    std::vector<float> A(NX * NY * sizeof(DATA_TYPE));
    std::vector<float> x1(NY);
    std::vector<float> x2(NX);
    std::vector<float> y1(NX, 0.0f);
    std::vector<float> y2(NY, 0.0f);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<float> distribution(0.0f, 40.0f);

	for(int j = 0; j < NY; j++)
		x1[j] = distribution(engine);

    for (int i = 0; i < NX; i++) {
		x2[i] = distribution(engine);
		for (j = 0; j < NY; j++)
			A[i*NY + j] = distribution(engine);
    }

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add two kernels to tuner, one of the kernels acts as reference kernel
    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "bicgFused", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId referenceKernelId = tuner.addKernelFromFile(referenceKernelFile, "bicgFusedRef", ndRangeDimensions,
        referenceWorkGroupDimensions);

    // Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers
    //tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", std::vector<size_t>{0, 1, 2, 4, 8, 16, 32});
    //tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", std::vector<size_t>{0, 1});
    //tuner.addParameter(kernelId, "VECTOR_TYPE", std::vector<size_t>{1, 2, 4, 8});
    //tuner.addParameter(kernelId, "USE_SOA", std::vector<size_t>{0, 1, 2});
	tuner.addParameter(kernelId, "BICG_BATCH", std::vector<size_t>{1, 2, 4, 8});

    // Using vectorized SoA only makes sense when vectors are longer than 1
    /*auto vectorizedSoA = [](std::vector<size_t> vector) {return vector.at(0) > 1 || vector.at(1) != 2;}; 
    tuner.addConstraint(kernelId, vectorizedSoA, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"});*/

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
    //tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", std::vector<size_t>{1, 2, 4, 8}, ktt::ModifierType::Global, ktt::ModifierAction::Divide,
        //ktt::ModifierDimension::X);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    //tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", std::vector<size_t>{4, 8, 16, 32}, ktt::ModifierType::Local, ktt::ModifierAction::Multiply,
    //    ktt::ModifierDimension::X);
    //tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", std::vector<size_t>{1, 2, 4, 8, 16, 32}, ktt::ModifierType::Local,
    //    ktt::ModifierAction::Multiply, ktt::ModifierDimension::Y);

    // Add all arguments utilized by kernels
    ktt::ArgumentId AId = tuner.addArgumentVector(A, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId x1Id = tuner.addArgumentVector(x1, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId x2Id = tuner.addArgumentVector(x2, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId y1Id = tuner.addArgumentVector(y1, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId y2Id = tuner.addArgumentVector(y2, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId mId = tuner.addArgumentScalar(m);
    ktt::ArgumentId nId = tuner.addArgumentScalar(n);
    //ktt::ArgumentId energyGridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
    tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{AId, x1Id, y1Id, x2Id, y2Id, mId, nId});
    tuner.setKernelArguments(referenceKernelId, std::vector<ktt::ArgumentId>{AId, x1Id, y1Id, x2Id, y2Id, mId, nId});

    // Set search method to random search, only 10% of all configurations will be explored.
    //tuner.setSearchMethod(ktt::SearchMethod::RandomSearch, std::vector<double>{0.1});

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.01);

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
    tuner.setReferenceKernel(kernelId, referenceKernelId, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{y1, y2});

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "bicg_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
