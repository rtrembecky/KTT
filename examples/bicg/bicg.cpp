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
#define N 8192 // 16384 // 32768
#define M 8192

/* Thread block dimensions */
#define WORK_GROUP_X 256
#define WORK_GROUP_Y 1

// New NVidia GPUs have max.workgroup size of 1024
#define MAX_WORK_GROUP_SIZE 1024

class BicgCpu : public ktt::ReferenceClass
{
public:
	BicgCpu(const ktt::ArgumentId arg1Id, const ktt::ArgumentId arg2Id, const std::vector<float>& A, const std::vector<float>& x1, const std::vector<float>& x2, const std::vector<float>& y1, const std::vector<float>& y2) :
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
		int i, j;

		for (i = 0; i < M; i++)
		{
			y2[i] = 0.0;
		}

		for (i = 0; i < N; i++)
		{
			y1[i] = 0.0;
			for (j = 0; j < M; j++)
			{
				y2[j] = y2[j] + x2[i] * A[i*M + j];
				y1[i] = y1[i] + A[i*M + j] * x1[j];
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
	const std::vector<float>& A;
	const std::vector<float>& x1;
	const std::vector<float>& x2;
	std::vector<float> y1;
	std::vector<float> y2;
};

class BicgManipulator : public ktt::TuningManipulator
{
public:
	BicgManipulator(const ktt::KernelId kernel1Id, const ktt::KernelId kernel2Id, const ktt::KernelId kernelFusedId) :
		kernel1Id(kernel1Id), kernel2Id(kernel2Id), kernelFusedId(kernelFusedId)
	{}

	// LaunchComputation is responsible for actual execution of tuned kernel
	void launchComputation(const ktt::KernelId kernelId) override
	{
		// Get kernel data
		ktt::DimensionVector globalSize = getCurrentGlobalSize(kernelFusedId);
		ktt::DimensionVector localSize = getCurrentLocalSize(kernelFusedId);

		std::vector<ktt::ParameterPair> parameterValues = getCurrentConfiguration();

		const int rowsProcessed = getParameterValue("ROWS_PROCESSED", parameterValues);
		const int tile = getParameterValue("TILE", parameterValues);
		const int bicgBatch = getParameterValue("BICG_BATCH", parameterValues);
		globalSize.setSizeX(M);
		globalSize.setSizeY(N / rowsProcessed * tile / bicgBatch);
		localSize.setSizeX(tile);
		localSize.setSizeY(tile / bicgBatch);
		printf("changed global to %d x %d and local to %d x %d\n", globalSize.getSizeX(), globalSize.getSizeY(), localSize.getSizeX(), localSize.getSizeY());

		if (getParameterValue("FUSED", parameterValues) == 1) {
			runKernel(kernelFusedId, globalSize, localSize);
		}
		else {
			runKernel(kernel1Id);
			runKernel(kernel2Id);
		}
	}

private:
	ktt::KernelId kernel1Id;
	ktt::KernelId kernel2Id;
	ktt::KernelId kernelFusedId;
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

	int by = N / 256; // unused

	// Declare kernel parameters
	const ktt::DimensionVector ndRangeDimensions(M, by * 32); // replaced in manipulator
	const ktt::DimensionVector workGroupDimensions(32, 32); // replaced in manipulator
	const ktt::DimensionVector referenceNdRangeDimensions1(ceil(N / WORK_GROUP_X)*WORK_GROUP_X, 1);
	const ktt::DimensionVector referenceNdRangeDimensions2(ceil(M / WORK_GROUP_X)*WORK_GROUP_X, 1);
	const ktt::DimensionVector referenceWorkGroupDimensions(WORK_GROUP_X, 1);

	// Declare data variables
	std::vector<float> A(N * M);
	std::vector<float> x1(M);
	std::vector<float> x2(N);
	std::vector<float> y1(N, 0.0f);
	std::vector<float> y2(M, 0.0f);

	// Initialize data
	std::random_device device;
	std::default_random_engine engine(device());
	std::uniform_real_distribution<float> distribution(0.0f, 100.0f);

	for (int j = 0; j < M; j++)
		x1[j] = distribution(engine); //1.0f;// (float)j; // distribution(engine);

	for (int i = 0; i < N; i++) {
		x2[i] = (float)i/1000;// 1.0f;// (float)i + 5;// distribution(engine);
		for (int j = 0; j < M; j++)
			A[i*M + j] = distribution(engine); //(float)(i + 1); //distribution(engine); // (j % 7 + 1) / 1000;// j > 1000 ? (float)j / 10000 : (float)(j + 1) / 1000;// distribution(engine);
		//if (i < 3) printf("A[%d*%d+[0,1,5]] = %.3f %.3f %.3f |", i, M, A[i*M + 0], A[i*M + 1], A[i*M + 5]);
	}

	// Create tuner object for specified platform and device
	ktt::Tuner tuner(platformIndex, deviceIndex);

	// Add two kernels to tuner, one of the kernels acts as reference kernel
	ktt::KernelId kernelFusedId = tuner.addKernelFromFile(kernelFile, "bicgFused", ndRangeDimensions, workGroupDimensions);
	ktt::KernelId kernel1Id = tuner.addKernelFromFile(referenceKernelFile, "bicgKernel1", referenceNdRangeDimensions1, referenceWorkGroupDimensions);
	ktt::KernelId kernel2Id = tuner.addKernelFromFile(referenceKernelFile, "bicgKernel2", referenceNdRangeDimensions2, referenceWorkGroupDimensions);
	ktt::KernelId kernelId = tuner.addComposition("BicgPolyBenchAndFused", std::vector<ktt::KernelId>{kernel1Id, kernel2Id, kernelFusedId}, std::make_unique<BicgManipulator>(kernel1Id, kernel2Id, kernelFusedId));
	
	// Add parameters to tuned kernel
	tuner.addParameter(kernelId, "FUSED", std::vector<size_t>{0, 1});
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "BICG_BATCH", std::vector<size_t>{ 1, 2, 4, 8}, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "USE_SHARED_MATRIX", std::vector<size_t>{ 0, 1 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "USE_SHARED_VECTOR1", std::vector<size_t>{ /*0,*/ 1 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "USE_SHARED_VECTOR2", std::vector<size_t>{ /*0,*/ 1 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "USE_SHARED_REDUCTION", std::vector<size_t>{ 0, 1 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "ATOMICS", std::vector<size_t>{ 1 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y); // TODO: add value 0 and reduce y1 and y2 on CPU
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "UNROLL_BICG_STEP", std::vector<size_t>{ /*0,*/ 1 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "ROWS_PROCESSED", std::vector<size_t>{ 128, 256, 512, 1024 }, ktt::ModifierType::None, ktt::ModifierAction::Divide, ktt::ModifierDimension::Y);
	tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "TILE", std::vector<size_t>{ 16, 32, 64 }, ktt::ModifierType::None, ktt::ModifierAction::Multiply, ktt::ModifierDimension::X);

	// All of the parameters are used only in the fused kernel
	auto fused = [](std::vector<size_t> vector) {return vector.at(0) == 1 || (vector.at(0) == 0 && vector.at(1) == 4 && vector.at(2) == 1 && vector.at(3) == 1 && vector.at(4) == 1 && vector.at(5) == 1 && vector.at(6) == 1 && vector.at(7) == 1 && vector.at(8) == 256) && vector.at(9) == 32; };
	tuner.addConstraint(kernelId, fused, std::vector<std::string>{"FUSED", "BICG_BATCH", "USE_SHARED_MATRIX", "USE_SHARED_VECTOR1", "USE_SHARED_VECTOR2", "USE_SHARED_REDUCTION", "ATOMICS", "UNROLL_BICG_STEP", "ROWS_PROCESSED", "TILE"});
	// New NVidia GPUs have max. workgroup size of 1024, so   tile_x * tile_y <= 1024   ==>   tile_x * (tile_x / batch) <= 1024
	auto maxWgSize = [](std::vector<size_t> vector) {return vector.at(0) * vector.at(0) / vector.at(1) <= MAX_WORK_GROUP_SIZE; };
	tuner.addConstraint(kernelId, maxWgSize, std::vector<std::string>{"TILE", "BICG_BATCH"});

	// Divide NDRange in dimension x by OUTER_UNROLL_FACTOR
	//tuner.addParameter(kernelId, "OUTER_UNROLL_FACTOR", std::vector<size_t>{1, 2, 4, 8}, ktt::ModifierType::Global, ktt::ModifierAction::Divide, ktt::ModifierDimension::X);

	// Multiply workgroup size in dimensions x and y by two parameters that follow (effectively setting workgroup size to parameters' values)
	//tuner.addCompositionKernelParameter(kernelId, kernelFusedId, "BATCH_X", std::vector<size_t>{ 1}, ktt::ModifierType::Both, ktt::ModifierAction::Divide, ktt::ModifierDimension::X);
	//tuner.addParameter(kernelId, "TILE_Y", std::vector<size_t>{1, 2, 4, 8, 16, 32}, ktt::ModifierType::Local, ktt::ModifierAction::Multiply, ktt::ModifierDimension::Y);

	// Add all arguments utilized by kernels
	ktt::ArgumentId AId = tuner.addArgumentVector(A, ktt::ArgumentAccessType::ReadOnly);
	ktt::ArgumentId x1Id = tuner.addArgumentVector(x1, ktt::ArgumentAccessType::ReadOnly);
	ktt::ArgumentId x2Id = tuner.addArgumentVector(x2, ktt::ArgumentAccessType::ReadOnly);
	ktt::ArgumentId y1Id = tuner.addArgumentVector(y1, ktt::ArgumentAccessType::ReadWrite);
	ktt::ArgumentId y2Id = tuner.addArgumentVector(y2, ktt::ArgumentAccessType::ReadWrite);
	ktt::ArgumentId mId = tuner.addArgumentScalar(M);
	ktt::ArgumentId mRefId = tuner.addArgumentScalar(M);
	ktt::ArgumentId nId = tuner.addArgumentScalar(N / by); // 256
	ktt::ArgumentId nRefId = tuner.addArgumentScalar(N);
	//ktt::ArgumentId energyGridId = tuner.addArgumentVector(energyGrid, ktt::ArgumentAccessType::ReadWrite);

	// Set kernel arguments for both tuned kernel and reference kernel, order of arguments is important
	tuner.setCompositionKernelArguments(kernelId, kernelFusedId, std::vector<ktt::ArgumentId>{AId, x1Id, y1Id, x2Id, y2Id, mId, nId});
	tuner.setCompositionKernelArguments(kernelId, kernel1Id, std::vector<ktt::ArgumentId>{AId, x1Id, y1Id, mRefId, nRefId});
	tuner.setCompositionKernelArguments(kernelId, kernel2Id, std::vector<ktt::ArgumentId>{AId, x2Id, y2Id, mRefId, nRefId});

	// Set search method to random search, only 10% of all configurations will be explored.
	//tuner.setSearchMethod(ktt::SearchMethod::RandomSearch, std::vector<double>{0.1});

	// Specify custom tolerance threshold for validation of floating point arguments. Default threshold is 1e-4.
	tuner.setValidationMethod(ktt::ValidationMethod::SideBySideRelativeComparison, 0.001);
	//tuner.setValidationMethod(ktt::ValidationMethod::SideBySideComparison, 0.0001);
	//tuner.setValidationRange(y2Id, 10);

	// Set tuning manipulator, which implements custom method for launching the kernel
	tuner.setTuningManipulator(kernelId, std::make_unique<BicgManipulator>(kernel1Id, kernel2Id, kernelFusedId));

	// Set reference kernel which validates results provided by tuned kernel, provide list of arguments which will be validated
	tuner.setReferenceClass(kernelId, std::make_unique<BicgCpu>(y1Id, y2Id, A, x1, x2, y1, y2), std::vector<ktt::ArgumentId>{y1Id, y2Id});

	// Launch kernel tuning
	tuner.tuneKernel(kernelId);

	// Print tuning results to standard output and to output.csv file
	tuner.printResult(kernelId, std::cout, ktt::PrintFormat::Verbose);
	tuner.printResult(kernelId, "bicg_output.csv", ktt::PrintFormat::CSV);

	return 0;
}
