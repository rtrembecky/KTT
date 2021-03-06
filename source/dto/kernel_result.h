#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "kernel/kernel_configuration.h"

namespace ktt
{

class KernelResult
{
public:
    KernelResult();
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration);
    explicit KernelResult(const std::string& kernelName, uint64_t computationDuration);
    explicit KernelResult(const std::string& kernelName, const KernelConfiguration& configuration, const std::string& errorMessage);

    void setKernelName(const std::string& kernelName);
    void setConfiguration(const KernelConfiguration& configuration);
    void setComputationDuration(const uint64_t computationDuration);
    void setOverhead(const uint64_t overhead);
    void setErrorMessage(const std::string& errorMessage);
    void setValid(const bool flag);

    std::string getKernelName() const;
    KernelConfiguration getConfiguration() const;
    uint64_t getComputationDuration() const;
    uint64_t getOverhead() const;
    std::string getErrorMessage() const;
    bool isValid() const;

    void increaseOverhead(const uint64_t overhead);

private:
    std::string kernelName;
    KernelConfiguration configuration;
    uint64_t computationDuration;
    uint64_t overhead;
    std::string errorMessage;
    bool valid;
};

} // namespace ktt
