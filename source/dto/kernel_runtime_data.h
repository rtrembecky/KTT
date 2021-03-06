#pragma once

#include <string>
#include <vector>
#include "ktt_types.h"
#include "api/dimension_vector.h"
#include "dto/local_memory_modifier.h"
#include "kernel_argument/kernel_argument.h"

namespace ktt
{

class KernelRuntimeData
{
public:
    explicit KernelRuntimeData(const KernelId id, const std::string& name, const std::string& source, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<ArgumentId>& argumentIds);
    explicit KernelRuntimeData(const KernelId id, const std::string& name, const std::string& source, const DimensionVector& globalSize,
        const DimensionVector& localSize, const std::vector<ArgumentId>& argumentIds, const std::vector<LocalMemoryModifier>& localMemoryModifiers);

    void setGlobalSize(const DimensionVector& globalSize);
    void setLocalSize(const DimensionVector& localSize);
    void setArgumentIndices(const std::vector<ArgumentId>& argumentIds);

    KernelId getId() const;
    std::string getName() const;
    std::string getSource() const;
    std::vector<size_t> getGlobalSize() const;
    std::vector<size_t> getLocalSize() const;
    DimensionVector getGlobalSizeDimensionVector() const;
    DimensionVector getLocalSizeDimensionVector() const;
    std::vector<ArgumentId> getArgumentIds() const;
    std::vector<LocalMemoryModifier> getLocalMemoryModifiers() const;

private:
    KernelId id;
    std::string name;
    std::string source;
    std::vector<size_t> globalSize;
    std::vector<size_t> localSize;
    DimensionVector globalSizeDimensionVector;
    DimensionVector localSizeDimensionVector;
    std::vector<ArgumentId> argumentIds;
    std::vector<LocalMemoryModifier> localMemoryModifiers;
};

} // namespace ktt
