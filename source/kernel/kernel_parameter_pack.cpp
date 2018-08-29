#include "kernel_parameter_pack.h"

namespace ktt
{

KernelParameterPack::KernelParameterPack() :
    name("")
{}

KernelParameterPack::KernelParameterPack(const std::string& name, const std::vector<std::string>& parameterNames) :
    name(name),
    parameterNames(parameterNames)
{}

std::string KernelParameterPack::getName() const
{
    return name;
}

std::vector<std::string> KernelParameterPack::getParameterNames() const
{
    return parameterNames;
}

bool KernelParameterPack::operator==(const KernelParameterPack& other) const
{
    return name == other.name;
}

bool KernelParameterPack::operator!=(const KernelParameterPack& other) const
{
    return !(*this == other);
}

} // namespace ktt