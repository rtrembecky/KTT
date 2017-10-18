#include <fstream>
#include <sstream>

#include "kernel_manager.h"
#include "utility/ktt_utility.h"

namespace ktt
{

KernelManager::KernelManager() :
    nextId(0),
    globalSizeType(GlobalSizeType::Opencl)
{}

size_t KernelManager::addKernel(const std::string& source, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    DimensionVector convertedGlobalSize = globalSize;

    if (globalSizeType == GlobalSizeType::Cuda)
    {
        convertedGlobalSize = DimensionVector(std::get<0>(globalSize) * std::get<0>(localSize), std::get<1>(globalSize) * std::get<1>(localSize),
            std::get<2>(globalSize) * std::get<2>(localSize));
    }

    kernels.emplace_back(nextId, source, kernelName, convertedGlobalSize, localSize);
    return nextId++;
}

size_t KernelManager::addKernelFromFile(const std::string& filePath, const std::string& kernelName, const DimensionVector& globalSize,
    const DimensionVector& localSize)
{
    std::string source = loadFileToString(filePath);
    return addKernel(source, kernelName, globalSize, localSize);
}

size_t KernelManager::addKernelComposition(const std::string& compositionName, const std::vector<size_t>& kernelIds)
{
    if (!containsUnique(kernelIds))
    {
        throw std::runtime_error("Kernels added to kernel composition must be unique");
    }

    std::vector<const Kernel*> compositionKernels;
    for (const auto& id : kernelIds)
    {
        if (!isKernel(id))
        {
            throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
        }
        compositionKernels.push_back(&kernels.at(id));
    }

    kernelCompositions.emplace_back(nextId, compositionName, compositionKernels);
    return nextId++;
}

std::string KernelManager::getKernelSourceWithDefines(const size_t id, const KernelConfiguration& kernelConfiguration) const
{
    std::string source = getKernel(id).getSource();

    for (const auto& parameterValue : kernelConfiguration.getParameterValues())
    {
        std::stringstream stream;
        stream << std::get<1>(parameterValue); // clean way to convert number to string
        source = std::string("#define ") + std::get<0>(parameterValue) + std::string(" ") + stream.str() + std::string("\n") + source;
    }

    return source;
}

KernelConfiguration KernelManager::getKernelConfiguration(const size_t id, const std::vector<ParameterValue>& parameterValues) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    const Kernel& kernel = getKernel(id);
    DimensionVector global = kernel.getGlobalSize();
    DimensionVector local = kernel.getLocalSize();
    
    for (const auto& value : parameterValues)
    {
        const KernelParameter* targetParameter = nullptr;
        for (const auto& parameter : kernel.getParameters())
        {
            if (parameter.getName() == std::get<0>(value))
            {
                targetParameter = &parameter;
                break;
            }
        }

        if (targetParameter == nullptr)
        {
            throw std::runtime_error(std::string("Parameter with name <") + std::get<0>(value) + "> is not associated with kernel with id: "
                + std::to_string(id));
        }

        global = modifyDimensionVector(global, DimensionVectorType::Global, *targetParameter, std::get<1>(value));
        local = modifyDimensionVector(local, DimensionVectorType::Local, *targetParameter, std::get<1>(value));
    }

    return KernelConfiguration(global, local, parameterValues);
}

KernelConfiguration KernelManager::getKernelCompositionConfiguration(const size_t compositionId,
    const std::vector<ParameterValue>& parameterValues) const
{
    if (!isKernelComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    const KernelComposition& kernelComposition = getKernelComposition(compositionId);
    std::vector<std::pair<size_t, DimensionVector>> globalSizes;
    std::vector<std::pair<size_t, DimensionVector>> localSizes;

    for (const auto& kernel : kernelComposition.getKernels())
    {
        globalSizes.push_back(std::make_pair(kernel->getId(), kernel->getGlobalSize()));
        localSizes.push_back(std::make_pair(kernel->getId(), kernel->getLocalSize()));
    }
    
    for (const auto& value : parameterValues)
    {
        const KernelParameter* targetParameter = nullptr;
        for (const auto& parameter : kernelComposition.getParameters())
        {
            if (parameter.getName() == std::get<0>(value))
            {
                targetParameter = &parameter;
                break;
            }
        }

        if (targetParameter == nullptr)
        {
            throw std::runtime_error(std::string("Parameter with name <") + std::get<0>(value)
                + "> is not associated with kernel composition with id: " + std::to_string(compositionId));
        }

        for (auto& globalSizePair : globalSizes)
        {
            for (const auto kernelId : targetParameter->getCompositionKernels())
            {
                if (globalSizePair.first == kernelId)
                {
                    globalSizePair.second = modifyDimensionVector(globalSizePair.second, DimensionVectorType::Global, *targetParameter,
                        std::get<1>(value));
                }
            }
        }
        for (auto& localSizePair : localSizes)
        {
            for (const auto kernelId : targetParameter->getCompositionKernels())
            {
                if (localSizePair.first == kernelId)
                {
                    localSizePair.second = modifyDimensionVector(localSizePair.second, DimensionVectorType::Local, *targetParameter,
                        std::get<1>(value));
                }
            }
        }
    }

    return KernelConfiguration(globalSizes, localSizes, parameterValues);
}

std::vector<KernelConfiguration> KernelManager::getKernelConfigurations(const size_t id, const DeviceInfo& deviceInfo) const
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }

    std::vector<KernelConfiguration> configurations;
    const Kernel& kernel = getKernel(id);

    if (kernel.getParameters().size() == 0)
    {
        configurations.emplace_back(kernel.getGlobalSize(), kernel.getLocalSize(), std::vector<ParameterValue>{});
    }
    else
    {
        computeConfigurations(0, deviceInfo, kernel.getParameters(), kernel.getConstraints(), std::vector<ParameterValue>(0), kernel.getGlobalSize(),
            kernel.getLocalSize(), configurations);
    }
    return configurations;
}

std::vector<KernelConfiguration> KernelManager::getKernelCompositionConfigurations(const size_t compositionId, const DeviceInfo& deviceInfo) const
{
    if (!isKernelComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    const KernelComposition& composition = getKernelComposition(compositionId);
    std::vector<std::pair<size_t, DimensionVector>> globalSizes;
    std::vector<std::pair<size_t, DimensionVector>> localSizes;

    for (const auto& kernel : composition.getKernels())
    {
        globalSizes.push_back(std::make_pair(kernel->getId(), kernel->getGlobalSize()));
        localSizes.push_back(std::make_pair(kernel->getId(), kernel->getLocalSize()));
    }

    std::vector<KernelConfiguration> kernelConfigurations;
    if (composition.getParameters().size() == 0)
    {
        kernelConfigurations.emplace_back(globalSizes, localSizes, std::vector<ParameterValue>{});
    }
    else
    {
        computeCompositionConfigurations(0, deviceInfo, composition.getParameters(), composition.getConstraints(), std::vector<ParameterValue>(0),
            globalSizes, localSizes, kernelConfigurations);
    }

    return kernelConfigurations;
}

void KernelManager::setGlobalSizeType(const GlobalSizeType& globalSizeType)
{
    this->globalSizeType = globalSizeType;
}

void KernelManager::addParameter(const size_t id, const std::string& name, const std::vector<size_t>& values,
    const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction, const Dimension& modifierDimension)
{
    if (isKernel(id))
    {
        getKernel(id).addParameter(KernelParameter(name, values, threadModifierType, threadModifierAction, modifierDimension));
    }
    else if (isKernelComposition(id))
    {
        getKernelComposition(id).addParameter(KernelParameter(name, values, threadModifierType, threadModifierAction, modifierDimension));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::addConstraint(const size_t id, const std::function<bool(std::vector<size_t>)>& constraintFunction,
    const std::vector<std::string>& parameterNames)
{
    if (isKernel(id))
    {
        getKernel(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
    }
    else if (isKernelComposition(id))
    {
        getKernelComposition(id).addConstraint(KernelConstraint(constraintFunction, parameterNames));
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::setArguments(const size_t id, const std::vector<size_t>& argumentIndices)
{
    if (isKernel(id))
    {
        getKernel(id).setArguments(argumentIndices);
    }
    else if (isKernelComposition(id))
    {
        getKernelComposition(id).setSharedArguments(argumentIndices);
    }
    else
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
}

void KernelManager::setTuningManipulatorFlag(const size_t id, const bool tuningManipulatorFlag)
{
    if (!isKernel(id))
    {
        throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
    }
    getKernel(id).setTuningManipulatorFlag(tuningManipulatorFlag);
}

void KernelManager::addCompositionKernelParameter(const size_t compositionId, const size_t kernelId, const std::string& parameterName,
    const std::vector<size_t>& parameterValues, const ThreadModifierType& threadModifierType, const ThreadModifierAction& threadModifierAction,
    const Dimension& modifierDimension)
{
    if (!isKernelComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    getKernelComposition(kernelId).addKernelParameter(kernelId, KernelParameter(parameterName, parameterValues, threadModifierType,
        threadModifierAction, modifierDimension));
}

void KernelManager::setCompositionKernelArguments(const size_t compositionId, const size_t kernelId, const std::vector<size_t>& argumentIds)
{
    if (!isKernelComposition(compositionId))
    {
        throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(compositionId));
    }

    getKernelComposition(compositionId).setKernelArguments(kernelId, argumentIds);
}

const Kernel& KernelManager::getKernel(const size_t id) const
{
    for (const auto& kernel : kernels)
    {
        if (kernel.getId() == id)
        {
            return kernel;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel id: ") + std::to_string(id));
}

Kernel& KernelManager::getKernel(const size_t id)
{
    return const_cast<Kernel&>(static_cast<const KernelManager*>(this)->getKernel(id));
}

size_t KernelManager::getCompositionCount() const
{
    return kernelCompositions.size();
}

const KernelComposition& KernelManager::getKernelComposition(const size_t id) const
{
    for (const auto& kernelComposition : kernelCompositions)
    {
        if (kernelComposition.getId() == id)
        {
            return kernelComposition;
        }
    }

    throw std::runtime_error(std::string("Invalid kernel composition id: ") + std::to_string(id));
}

KernelComposition& KernelManager::getKernelComposition(const size_t id)
{
    return const_cast<KernelComposition&>(static_cast<const KernelManager*>(this)->getKernelComposition(id));
}

bool KernelManager::isKernel(const size_t id) const
{
    for (const auto& kernel : kernels)
    {
        if (kernel.getId() == id)
        {
            return true;
        }
    }

    return false;
}

bool KernelManager::isKernelComposition(const size_t id) const
{
    for (const auto& kernelComposition : kernelCompositions)
    {
        if (kernelComposition.getId() == id)
        {
            return true;
        }
    }

    return false;
}

std::string KernelManager::loadFileToString(const std::string& filePath) const
{
    std::ifstream file(filePath);

    if (!file.is_open())
    {
        throw std::runtime_error(std::string("Unable to open file: ") + filePath);
    }

    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
}

void KernelManager::computeConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo,
    const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
    const std::vector<ParameterValue>& parameterValues, const DimensionVector& globalSize, const DimensionVector& localSize,
    std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSize, localSize, parameterValues);
        if (configurationIsValid(configuration, constraints, deviceInfo))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter
    for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
    {
        auto newParameterValues = parameterValues;
        newParameterValues.push_back(ParameterValue(parameter.getName(), value));

        auto newGlobalSize = modifyDimensionVector(globalSize, DimensionVectorType::Global, parameter, value);
        auto newLocalSize = modifyDimensionVector(localSize, DimensionVectorType::Local, parameter, value);

        computeConfigurations(currentParameterIndex + 1, deviceInfo, parameters, constraints, newParameterValues, newGlobalSize, newLocalSize,
            finalResult);
    }
}

void KernelManager::computeCompositionConfigurations(const size_t currentParameterIndex, const DeviceInfo& deviceInfo,
    const std::vector<KernelParameter>& parameters, const std::vector<KernelConstraint>& constraints,
    const std::vector<ParameterValue>& parameterValues, std::vector<std::pair<size_t, DimensionVector>>& globalSizes,
    std::vector<std::pair<size_t, DimensionVector>>& localSizes, std::vector<KernelConfiguration>& finalResult) const
{
    if (currentParameterIndex >= parameters.size()) // all parameters are now part of the configuration
    {
        KernelConfiguration configuration(globalSizes, localSizes, parameterValues);
        if (configurationIsValid(configuration, constraints, deviceInfo))
        {
            finalResult.push_back(configuration);
        }
        return;
    }

    KernelParameter parameter = parameters.at(currentParameterIndex); // process next parameter
    for (const auto& value : parameter.getValues()) // recursively build tree of configurations for each parameter value
    {
        auto newParameterValues = parameterValues;
        newParameterValues.push_back(ParameterValue(parameter.getName(), value));

        for (const auto compositionKernelId : parameter.getCompositionKernels())
        {
            for (auto& globalSizePair : globalSizes)
            {
                if (compositionKernelId == globalSizePair.first)
                {
                    DimensionVector newGlobalSize = modifyDimensionVector(globalSizePair.second, DimensionVectorType::Global, parameter, value);
                    globalSizePair.second = newGlobalSize;
                }
            }

            for (auto& localSizePair : localSizes)
            {
                if (compositionKernelId == localSizePair.first)
                {
                    DimensionVector newLocalSize = modifyDimensionVector(localSizePair.second, DimensionVectorType::Local, parameter, value);
                    localSizePair.second = newLocalSize;
                }
            }
        }

        computeCompositionConfigurations(currentParameterIndex + 1, deviceInfo, parameters, constraints, newParameterValues, globalSizes, localSizes,
            finalResult);
    }
}

DimensionVector KernelManager::modifyDimensionVector(const DimensionVector& vector, const DimensionVectorType& dimensionVectorType,
    const KernelParameter& parameter, const size_t parameterValue) const
{
    if (parameter.getThreadModifierType() == ThreadModifierType::None
        || dimensionVectorType == DimensionVectorType::Global && parameter.getThreadModifierType() == ThreadModifierType::Local
        || dimensionVectorType == DimensionVectorType::Local && parameter.getThreadModifierType() == ThreadModifierType::Global)
    {
        return vector;
    }

    ThreadModifierAction action = parameter.getThreadModifierAction();
    if (action == ThreadModifierAction::Add)
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) + parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) + parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) + parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
    else if (action == ThreadModifierAction::Subtract)
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) - parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) - parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) - parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
    else if (action == ThreadModifierAction::Multiply)
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) * parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) * parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) * parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
    else // divide
    {
        size_t x = parameter.getModifierDimension() == Dimension::X ? std::get<0>(vector) / parameterValue : std::get<0>(vector);
        size_t y = parameter.getModifierDimension() == Dimension::Y ? std::get<1>(vector) / parameterValue : std::get<1>(vector);
        size_t z = parameter.getModifierDimension() == Dimension::Z ? std::get<2>(vector) / parameterValue : std::get<2>(vector);
        return DimensionVector(x, y, z);
    }
}

bool KernelManager::configurationIsValid(const KernelConfiguration& configuration, const std::vector<KernelConstraint>& constraints,
    const DeviceInfo& deviceInfo) const
{
    for (const auto& constraint : constraints)
    {
        std::vector<std::string> constraintNames = constraint.getParameterNames();
        auto constraintValues = std::vector<size_t>(constraintNames.size());

        for (size_t i = 0; i < constraintNames.size(); i++)
        {
            for (const auto& parameterValue : configuration.getParameterValues())
            {
                if (std::get<0>(parameterValue) == constraintNames.at(i))
                {
                    constraintValues.at(i) = std::get<1>(parameterValue);
                    break;
                }
            }
        }

        auto constraintFunction = constraint.getConstraintFunction();
        if (!constraintFunction(constraintValues))
        {
            return false;
        }
    }

    auto localSizes = configuration.getLocalSizes();
    for (const auto& localSize : localSizes)
    {
        if (std::get<0>(localSize) * std::get<1>(localSize) * std::get<2>(localSize) > deviceInfo.getMaxWorkGroupSize())
        {
            return false;
        }
    }

    return true;
}

} // namespace ktt
