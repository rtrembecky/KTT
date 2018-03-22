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

/* Problem size. */
#define NX 4096
#define NY 4096

/* Thread block dimensions */
#define WORK_GROUP_X 256
#define WORK_GROUP_Y 1

typedef float DATA_TYPE;

class SimpleValidator : public ktt::ReferenceClass
{
public:
    SimpleValidator(const ktt::ArgumentId arg1Id, const ktt::ArgumentId arg2Id, const std::vector<DATA_TYPE>& A, const std::vector<DATA_TYPE>& x1, const std::vector<DATA_TYPE>& x2, const std::vector<DATA_TYPE>& y1, const std::vector<DATA_TYPE>& y2) :
        arg1Id(arg1Id),
        arg2Id(arg2Id),
        A(A),
	x1(x1),
	x2(x2),
	y1(y1),
	y2(y2)
    {}

    // Method inherited from ReferenceClass, which computes reference result for all arguments that are validated inside the class.
    void computeResult() override
    {
        int i,j;
	
  	for (i = 0; i < NY; i++)
	{
		y2[i] = 0.0;
	}

	for (i = 0; i < NX; i++)
	{
		y1[i] = 0.0;
		for (j = 0; j < NY; j++)
	  	{
	    		y2[j] = y2[j] + x2[i] * A[i*NY + j];
	    		y1[i] = y1[i] + A[i*NY + j] * x1[j];
	  	}
	}
    }

    // Method inherited from ReferenceClass, which returns memory location where reference result for corresponding argument is stored.
    void* getData(const ktt::ArgumentId id) override
    {
        if (id == arg1Id)
        {
            return y1.data();
        }
        if (id == arg2Id)
        {
            return y2.data();
        }
        return nullptr;
    }

private:
    ktt::ArgumentId arg1Id;
    ktt::ArgumentId arg2Id;
    const std::vector<DATA_TYPE>& A;
    const std::vector<DATA_TYPE>& x1;
    const std::vector<DATA_TYPE>& x2;
    std::vector<DATA_TYPE> y1;
    std::vector<DATA_TYPE> y2;
};

class BicgManipulator : public ktt::TuningManipulator
{
public:
    BicgManipulator(const ktt::KernelId kernel1Id, const ktt::KernelId kernel2Id) :
	kernel1Id(kernel1Id), kernel2Id(kernel2Id)
    {}

    // LaunchComputation is responsible for actual execution of tuned kernel
    void launchComputation(const ktt::KernelId kernelId) override
    {
        // Get kernel data
        ktt::DimensionVector globalSize = getCurrentGlobalSize(kernel1Id);
        ktt::DimensionVector localSize = getCurrentLocalSize(kernel1Id);
        std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

        runKernel(kernel1Id);
	runKernel(kernel2Id);
	}

private:
	ktt::KernelId kernel1Id;
	ktt::KernelId kernel2Id;
    int atoms;
    int gridSize;
    float gridSpacing;
    ktt::ArgumentId atomInfoPrecompId;
    ktt::ArgumentId atomInfoZ2Id;
    ktt::ArgumentId zIndexId;
    std::vector<float> atomInfoPrecomp;
    std::vector<float> atomInfoZ;
    std::vector<float> atomInfoZ2;
};

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

	int by = NX/256;

    // Declare kernel parameters
    const ktt::DimensionVector ndRangeDimensions(NY, by*32);
    const ktt::DimensionVector referenceNdRangeDimensions1(ceil(NX/WORK_GROUP_X)*WORK_GROUP_X, 1);
    const ktt::DimensionVector referenceNdRangeDimensions2(ceil(NY/WORK_GROUP_X)*WORK_GROUP_X, 1);
    const ktt::DimensionVector workGroupDimensions(32, 32);
    const ktt::DimensionVector referenceWorkGroupDimensions(WORK_GROUP_X, 1);

    // Declare data variables
    std::vector<DATA_TYPE> A(NX * NY);
    std::vector<DATA_TYPE> x1(NY);
    std::vector<DATA_TYPE> x2(NX);
    std::vector<DATA_TYPE> y1(NX, 0.0f);
    std::vector<DATA_TYPE> y2(NY, 0.0f);

    // Initialize data
    std::random_device device;
    std::default_random_engine engine(device());
    std::uniform_real_distribution<DATA_TYPE> distribution(0.0f, 40.0f);

	for(int j = 0; j < NY; j++)
		x1[j] = distribution(engine);

    	for (int i = 0; i < NX; i++) {
		x2[i] = distribution(engine);
		for (int j = 0; j < NY; j++)
			A[i*NY + j] = distribution(engine);
	}

    // Create tuner object for specified platform and device
    ktt::Tuner tuner(platformIndex, deviceIndex);

    // Add two kernels to tuner, one of the kernels acts as reference kernel
//    ktt::KernelId kernelId = tuner.addKernelFromFile(kernelFile, "bicgFused", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId kernel1Id = tuner.addKernelFromFile(referenceKernelFile, "bicgKernel1", referenceNdRangeDimensions1, referenceWorkGroupDimensions);
    ktt::KernelId kernel2Id = tuner.addKernelFromFile(referenceKernelFile, "bicgKernel2", referenceNdRangeDimensions2, referenceWorkGroupDimensions);
	ktt::KernelId kernelId = tuner.addComposition("BicgPolyBench", std::vector<ktt::KernelId>{kernel1Id, kernel2Id}, std::make_unique<BicgManipulator>(kernel1Id, kernel2Id));
    // Add several parameters to tuned kernel, some of them utilize constraint function and thread modifiers
    //tuner.addParameter(kernelId, "INNER_UNROLL_FACTOR", std::vector<size_t>{0, 1, 2, 4, 8, 16, 32});
    //tuner.addParameter(kernelId, "USE_CONSTANT_MEMORY", std::vector<size_t>{0, 1});
    //tuner.addParameter(kernelId, "VECTOR_TYPE", std::vector<size_t>{1, 2, 4, 8});
    //tuner.addParameter(kernelId, "USE_SOA", std::vector<size_t>{0, 1, 2});
	tuner.addParameter(kernelId, "BICG_BATCH", std::vector<size_t>{1, 2, 4, 8}, ktt::ModifierType::Both, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addParameter(kernel2Id, "BICG", std::vector<size_t>{1}, ktt::ModifierType::Both, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
    // Using vectorized SoA only makes sense when vectors are longer than 1
    /*auto vectorizedSoA = [](std::vector<size_t> vector) {return vector.at(0) > 1 || vector.at(1) != 2;}; 
    tuner.addConstraint(kernelId, vectorizedSoA, std::vector<std::string>{"VECTOR_TYPE", "USE_SOA"});*/

    // Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
    //tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", std::vector<size_t>{1, 2, 4, 8}, ktt::ModifierType::Global, ktt::ModifierAction::Divide, ktt::ModifierDimension::X);

    // Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
    //tuner.addParameter(kernelId, "WORK_GROUP_SIZE_X", std::vector<size_t>{4, 8, 16, 32}, ktt::ModifierType::Local, ktt::ModifierAction::Multiply, ktt::ModifierDimension::X);
    //tuner.addParameter(kernelId, "WORK_GROUP_SIZE_Y", std::vector<size_t>{1, 2, 4, 8, 16, 32}, ktt::ModifierType::Local, ktt::ModifierAction::Multiply, ktt::ModifierDimension::Y);

    // Add all arguments utilized by kernels
    ktt::ArgumentId AId = tuner.addArgumentVector(A, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId x1Id = tuner.addArgumentVector(x1, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId x2Id = tuner.addArgumentVector(x2, ktt::ArgumentAccessType::ReadOnly);
    ktt::ArgumentId y1Id = tuner.addArgumentVector(y1, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId y2Id = tuner.addArgumentVector(y2, ktt::ArgumentAccessType::ReadWrite);
    ktt::ArgumentId mId = tuner.addArgumentScalar(NX/by);
    ktt::ArgumentId mRefId = tuner.addArgumentScalar(NX);
    ktt::ArgumentId nId = tuner.addArgumentScalar(NY);
    //ktt::ArgumentId energyGridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

    // Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
	//tuner.setKernelArguments(kernelId, std::vector<ktt::ArgumentId>{AId, x1Id, y1Id, x2Id, y2Id, mId, nId});
	tuner.setCompositionKernelArguments(kernelId, kernel1Id, std::vector<ktt::ArgumentId>{AId, x1Id, y1Id, mRefId, nId});
	tuner.setCompositionKernelArguments(kernelId, kernel2Id, std::vector<ktt::ArgumentId>{AId, x2Id, y2Id, mRefId, nId});

    // Set search method to random search, only 10% of all configurations will be explored.
    //tuner.setSearchMethod(ktt::SearchMethod::RandomSearch, std::vector<double>{0.1});

    // Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
    //tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 2.0);

    // Set tuning manipulator, which implements custom method for launching the kernel
	tuner.setTuningManipulator(kernelId, std::make_unique<BicgManipulator>(kernel1Id, kernel2Id));

    // Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
	//tuner.setReferenceKernel(kernelId, kernel1Id, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{y1Id});
	//tuner.setReferenceKernel(kernelId, kernel2Id, std::vector<ktt::ParameterPair>{}, std::vector<ktt::ArgumentId>{y2Id});
	tuner.setReferenceClass(kernelId, std::make_unique<SimpleValidator>(y1Id, y2Id, A, x1, x2, y1, y2), std::vector<ktt::ArgumentId>{y1Id, y2Id});

    // Launch kernel tuning
    tuner.tuneKernel(kernelId);

    // Print tuning results to standard output and to output.csv file
    tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
    tuner.printResult(kernelId, "bicg_output.csv", ktt::PrintFormat::CSV);

    return 0;
}
