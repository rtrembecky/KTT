#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "configuration_manager.h"
#include "kernel_runner.h"
#include "result_validator.h"
#include "dto/kernel_result.h"
#include "kernel/kernel_manager.h"
#include "kernel_argument/argument_manager.h"
#include "utility/logger.h"

namespace ktt
{

class TuningRunner
{
public:
    // Constructor
    explicit TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, KernelRunner* kernelRunner, Logger* logger);

    // Core methods
    std::vector<KernelResult> tuneKernel(const KernelId id);
    std::vector<KernelResult> dryTuneKernel(const KernelId id, const std::string& filePath);
    std::vector<KernelResult> tuneComposition(const KernelId id);
    KernelResult tuneKernelByStep(const KernelId id, const std::vector<OutputDescriptor>& output);
    KernelResult tuneCompositionByStep(const KernelId id, const std::vector<OutputDescriptor>& output);
    void setSearchMethod(const SearchMethod method, const std::vector<double>& arguments);
    void setValidationMethod(const ValidationMethod method, const double toleranceThreshold);
    void setValidationRange(const ArgumentId id, const size_t range);
    void setArgumentComparator(const ArgumentId id, const std::function<bool(const void*, const void*)>& comparator);
    void setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
        const std::vector<ArgumentId>& validatedArgumentIds);
    void setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass, const std::vector<ArgumentId>& validatedArgumentIds);
    std::vector<ParameterPair> getBestConfiguration(const KernelId id) const;

private:
    // Attributes
    ArgumentManager* argumentManager;
    KernelManager* kernelManager;
    KernelRunner* kernelRunner;
    Logger* logger;
    ConfigurationManager configurationManager;
    std::unique_ptr<ResultValidator> resultValidator;

    // Helper methods
    bool validateResult(const Kernel& kernel, const KernelResult& result);
    bool hasWritableZeroCopyArguments(const Kernel& kernel);
};

} // namespace ktt
