#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include "tuning_runner.h"
#include "searcher/annealing_searcher.h"
#include "searcher/full_searcher.h"
#include "searcher/pso_searcher.h"
#include "searcher/random_searcher.h"
#include "utility/ktt_utility.h"
#include "utility/timer.h"

namespace ktt
{

TuningRunner::TuningRunner(ArgumentManager* argumentManager, KernelManager* kernelManager, Logger* logger, ComputeEngine* computeEngine,
    const RunMode& runMode) :
    argumentManager(argumentManager),
    kernelManager(kernelManager),
    logger(logger),
    computeEngine(computeEngine),
    resultValidator(nullptr),
    manipulatorInterfaceImplementation(std::make_unique<ManipulatorInterfaceImplementation>(computeEngine)),
    searchMethod(SearchMethod::FullSearch),
    runMode(runMode)
{
    if (runMode == RunMode::Tuning)
    {
        resultValidator = std::make_unique<ResultValidator>(argumentManager, kernelManager, logger, computeEngine);
    }
}

std::vector<TuningResult> TuningRunner::tuneKernel(const KernelId id)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Kernel tuning cannot be performed in computation mode");
    }

    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results;
    const Kernel& kernel = kernelManager->getKernel(id);
    resultValidator->computeReferenceResult(kernel);

    std::unique_ptr<Searcher> searcher = getSearcher(searchMethod, searchArguments, kernelManager->getKernelConfigurations(id,
        computeEngine->getCurrentDeviceInfo()), kernel.getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        TuningResult result(kernel.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel <" << kernel.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount << "): "
                << currentConfiguration;
            logger->log(stream.str());

            if (kernel.hasTuningManipulator())
            {
                auto manipulatorPointer = tuningManipulators.find(id);
                result = runKernelWithManipulator(kernel, manipulatorPointer->second.get(), currentConfiguration,
                    std::vector<ArgumentOutputDescriptor>{});
            }
            else
            {
                result = runKernelSimple(kernel, currentConfiguration, std::vector<ArgumentOutputDescriptor>{});
            }
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
            results.emplace_back(kernel.getName(), currentConfiguration, std::string("Failed kernel run: ") + error.what());
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getTotalDuration()));
        if (validateResult(kernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(kernel.getName(), currentConfiguration, "Results differ");
        }

        computeEngine->clearBuffers(ArgumentAccessType::ReadWrite);
        computeEngine->clearBuffers(ArgumentAccessType::WriteOnly);

        if (kernel.hasTuningManipulator())
        {
            computeEngine->clearBuffers(ArgumentAccessType::ReadOnly);
        }
    }

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    return results;
}

std::vector<TuningResult> TuningRunner::tuneKernelComposition(const KernelId id)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Kernel tuning cannot be performed in computation mode");
    }

    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    std::vector<TuningResult> results;
    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const Kernel& compatibilityKernel = compositionToKernel(composition);
    resultValidator->computeReferenceResult(compatibilityKernel);

    std::unique_ptr<Searcher> searcher = getSearcher(searchMethod, searchArguments, kernelManager->getKernelCompositionConfigurations(id,
        computeEngine->getCurrentDeviceInfo()), composition.getParameters());
    size_t configurationsCount = searcher->getConfigurationsCount();

    for (size_t i = 0; i < configurationsCount; i++)
    {
        KernelConfiguration currentConfiguration = searcher->getNextConfiguration();
        TuningResult result(composition.getName(), currentConfiguration);

        try
        {
            std::stringstream stream;
            stream << "Launching kernel composition <" << composition.getName() << "> with configuration (" << i + 1 << " / " << configurationsCount
                << "): " << currentConfiguration;
            logger->log(stream.str());

            auto manipulatorPointer = tuningManipulators.find(id);
            result = runCompositionWithManipulator(composition, manipulatorPointer->second.get(), currentConfiguration,
                std::vector<ArgumentOutputDescriptor>{});
        }
        catch (const std::runtime_error& error)
        {
            logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
            results.emplace_back(composition.getName(), currentConfiguration, std::string("Failed kernel composition run: ") + error.what());
        }

        searcher->calculateNextConfiguration(static_cast<double>(result.getTotalDuration()));
        if (validateResult(compatibilityKernel, result))
        {
            results.push_back(result);
        }
        else
        {
            results.emplace_back(composition.getName(), currentConfiguration, "Results differ");
        }

        computeEngine->clearBuffers();
    }

    computeEngine->clearBuffers();
    resultValidator->clearReferenceResults();
    return results;
}

void TuningRunner::runKernel(const KernelId id, const std::vector<ParameterPair>& configuration, const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = kernelManager->getKernel(id);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelConfiguration(id, configuration);

    std::stringstream stream;
    stream << "Running kernel <" << kernel.getName() << "> with configuration: " << launchConfiguration;
    logger->log(stream.str());

    try
    {
        if (kernel.hasTuningManipulator())
        {
            auto manipulatorPointer = tuningManipulators.find(id);
            runKernelWithManipulator(kernel, manipulatorPointer->second.get(), launchConfiguration, output);
        }
        else
        {
            runKernelSimple(kernel, launchConfiguration, output);
        }
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
}

void TuningRunner::runComposition(const KernelId id, const std::vector<ParameterPair>& configuration,
    const std::vector<ArgumentOutputDescriptor>& output)
{
    if (!kernelManager->isComposition(id))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
    }

    const KernelComposition& composition = kernelManager->getKernelComposition(id);
    const KernelConfiguration launchConfiguration = kernelManager->getKernelCompositionConfiguration(id, configuration);

    std::stringstream stream;
    stream << "Running kernel composition <" << composition.getName() << "> with configuration: " << launchConfiguration;
    logger->log(stream.str());

    try
    {
        auto manipulatorPointer = tuningManipulators.find(id);
        runCompositionWithManipulator(composition, manipulatorPointer->second.get(), launchConfiguration, output);
    }
    catch (const std::runtime_error& error)
    {
        logger->log(std::string("Kernel composition run failed, reason: ") + error.what() + "\n");
    }

    computeEngine->clearBuffers();
}

void TuningRunner::setSearchMethod(const SearchMethod& method, const std::vector<double>& arguments)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Kernel tuning cannot be performed in computation mode");
    }

    if (method == SearchMethod::RandomSearch && arguments.size() < 1
        || method == SearchMethod::Annealing && arguments.size() < 2
        || method == SearchMethod::PSO && arguments.size() < 5)
    {
        throw std::runtime_error(std::string("Insufficient number of arguments given for specified search method: ")
            + getSearchMethodName(method));
    }
    
    this->searchArguments = arguments;
    this->searchMethod = method;
}

void TuningRunner::setValidationMethod(const ValidationMethod& method, const double toleranceThreshold)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setValidationMethod(method);
    resultValidator->setToleranceThreshold(toleranceThreshold);
}

void TuningRunner::setValidationRange(const ArgumentId id, const size_t range)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setValidationRange(id, range);
}

void TuningRunner::setReferenceKernel(const KernelId id, const KernelId referenceId, const std::vector<ParameterPair>& referenceConfiguration,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setReferenceKernel(id, referenceId, referenceConfiguration, validatedArgumentIds);
}

void TuningRunner::setReferenceClass(const KernelId id, std::unique_ptr<ReferenceClass> referenceClass,
    const std::vector<ArgumentId>& validatedArgumentIds)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }
    resultValidator->setReferenceClass(id, std::move(referenceClass), validatedArgumentIds);
}

void TuningRunner::setTuningManipulator(const KernelId id, std::unique_ptr<TuningManipulator> manipulator)
{
    if (tuningManipulators.find(id) != tuningManipulators.end())
    {
        tuningManipulators.erase(id);
    }
    tuningManipulators.insert(std::make_pair(id, std::move(manipulator)));
}

TuningResult TuningRunner::runKernelSimple(const Kernel& kernel, const KernelConfiguration& configuration,
    const std::vector<ArgumentOutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string kernelName = kernel.getName();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

    KernelRuntimeData kernelData(kernelId, kernelName, source, configuration.getGlobalSize(), configuration.getLocalSize(), kernel.getArgumentIds());
    KernelRunResult result = computeEngine->runKernel(kernelData, argumentManager->getArguments(kernel.getArgumentIds()), output);
    return TuningResult(kernelName, configuration, result);
}

TuningResult TuningRunner::runKernelWithManipulator(const Kernel& kernel, TuningManipulator* manipulator, const KernelConfiguration& configuration,
    const std::vector<ArgumentOutputDescriptor>& output)
{
    KernelId kernelId = kernel.getId();
    std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);
    KernelRuntimeData kernelData(kernelId, kernel.getName(), source, configuration.getGlobalSize(), configuration.getLocalSize(),
        kernel.getArgumentIds());

    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);
    manipulatorInterfaceImplementation->setConfiguration(configuration);
    manipulatorInterfaceImplementation->setKernelArguments(argumentManager->getArguments(kernel.getArgumentIds()));
    if (manipulator->enableArgumentPreload())
    {
        manipulatorInterfaceImplementation->uploadBuffers();
    }

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(kernelId);
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    TuningResult tuningResult(kernel.getName(), configuration, result);
    tuningResult.setManipulatorDuration(manipulatorDuration);
    return tuningResult;
}

TuningResult TuningRunner::runCompositionWithManipulator(const KernelComposition& composition, TuningManipulator* manipulator,
    const KernelConfiguration& configuration, const std::vector<ArgumentOutputDescriptor>& output)
{
    manipulator->manipulatorInterface = manipulatorInterfaceImplementation.get();
    std::vector<KernelArgument*> allArguments = argumentManager->getArguments(composition.getSharedArgumentIds());

    for (const auto kernel : composition.getKernels())
    {
        KernelId kernelId = kernel->getId();
        std::vector<ArgumentId> argumentIds = composition.getKernelArgumentIds(kernelId);
        std::string source = kernelManager->getKernelSourceWithDefines(kernelId, configuration);

        KernelRuntimeData kernelData(kernelId, kernel->getName(), source, configuration.getCompositionKernelGlobalSize(kernelId),
            configuration.getCompositionKernelLocalSize(kernelId), argumentIds);
        manipulatorInterfaceImplementation->addKernel(kernelId, kernelData);

        std::vector<KernelArgument*> newArguments = argumentManager->getArguments(argumentIds);
        for (const auto newArgument : newArguments)
        {
            if (!elementExists(newArgument, allArguments))
            {
                allArguments.push_back(newArgument);
            }
        }
    }

    manipulatorInterfaceImplementation->setConfiguration(configuration);
    manipulatorInterfaceImplementation->setKernelArguments(allArguments);
    if (manipulator->enableArgumentPreload())
    {
        manipulatorInterfaceImplementation->uploadBuffers();
    }

    Timer timer;
    try
    {
        timer.start();
        manipulator->launchComputation(composition.getId());
        timer.stop();
    }
    catch (const std::runtime_error&)
    {
        manipulatorInterfaceImplementation->clearData();
        manipulator->manipulatorInterface = nullptr;
        throw;
    }

    manipulatorInterfaceImplementation->downloadBuffers(output);
    KernelRunResult result = manipulatorInterfaceImplementation->getCurrentResult();
    size_t manipulatorDuration = timer.getElapsedTime();
    manipulatorDuration -= result.getOverhead();

    manipulatorInterfaceImplementation->clearData();
    manipulator->manipulatorInterface = nullptr;

    TuningResult tuningResult(composition.getName(), configuration, result);
    tuningResult.setManipulatorDuration(manipulatorDuration);
    return tuningResult;
}

std::unique_ptr<Searcher> TuningRunner::getSearcher(const SearchMethod& method, const std::vector<double>& arguments,
    const std::vector<KernelConfiguration>& configurations, const std::vector<KernelParameter>& parameters) const
{
    std::unique_ptr<Searcher> searcher;

    switch (method)
    {
    case SearchMethod::FullSearch:
        searcher = std::make_unique<FullSearcher>(configurations);
        break;
    case SearchMethod::RandomSearch:
        searcher = std::make_unique<RandomSearcher>(configurations, arguments.at(0));
        break;
    case SearchMethod::PSO:
        searcher = std::make_unique<PSOSearcher>(configurations, parameters, arguments.at(0), static_cast<size_t>(arguments.at(1)), arguments.at(2),
            arguments.at(3), arguments.at(4));
        break;
    case SearchMethod::Annealing:
        searcher = std::make_unique<AnnealingSearcher>(configurations, arguments.at(0), arguments.at(1));
        break;
    default:
        throw std::runtime_error("Specified searcher is not supported");
    }

    return searcher;
}

bool TuningRunner::validateResult(const Kernel& kernel, const TuningResult& result)
{
    if (runMode == RunMode::Computation)
    {
        throw std::runtime_error("Validation cannot be performed in computation mode");
    }

    if (!result.isValid())
    {
        return false;
    }

    bool resultIsCorrect = resultValidator->validateArgumentsWithClass(kernel, result.getConfiguration());
    resultIsCorrect &= resultValidator->validateArgumentsWithKernel(kernel, result.getConfiguration());

    if (resultIsCorrect)
    {
        logger->log(std::string("Kernel run completed successfully in ") + std::to_string((result.getTotalDuration()) / 1'000'000)
            + "ms\n");
    }
    else
    {
        logger->log("Kernel run completed successfully, but results differ\n");
    }

    return resultIsCorrect;
}

std::string TuningRunner::getSearchMethodName(const SearchMethod& method) const
{
    switch (method)
    {
    case SearchMethod::FullSearch:
        return std::string("FullSearch");
    case SearchMethod::RandomSearch:
        return std::string("RandomSearch");
    case SearchMethod::PSO:
        return std::string("PSO");
    case SearchMethod::Annealing:
        return std::string("Annealing");
    default:
        return std::string("Unknown search method");
    }
}

Kernel TuningRunner::compositionToKernel(const KernelComposition& composition) const
{
    Kernel kernel(composition.getId(), "", composition.getName(), DimensionVector(), DimensionVector());
    kernel.setTuningManipulatorFlag(true);

    for (const auto& constraint : composition.getConstraints())
    {
        kernel.addConstraint(constraint);
    }

    for (const auto& parameter : composition.getParameters())
    {
        kernel.addParameter(parameter);
    }

    std::vector<size_t> argumentIds;
    for (const auto id : composition.getSharedArgumentIds())
    {
        argumentIds.push_back(id);
    }

    for (const auto& kernel : composition.getKernels())
    {
        for (const auto id : composition.getKernelArgumentIds(kernel->getId()))
        {
            argumentIds.push_back(id);
        }
    }
    kernel.setArguments(argumentIds);

    return kernel;
}

} // namespace ktt
