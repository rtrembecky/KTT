#pragma once

#include "searcher.h"

namespace ktt
{

class FullSearcher : public Searcher
{
public:
    FullSearcher(const std::vector<KernelConfiguration>& configurations) :
        configurations(configurations),
        index(0)
    {
        if (configurations.size() == 0)
        {
            throw std::runtime_error("Configurations vector provided for searcher is empty");
        }
    }

    void calculateNextConfiguration(const double) override
    {
        index++;
    }

    KernelConfiguration getCurrentConfiguration() const override
    {
        return configurations.at(index);
    }

    size_t getConfigurationCount() const override
    {
        return configurations.size();
    }

    size_t getUnexploredConfigurationCount() const override
    {
        return getConfigurationCount() - index;
    }

private:
    std::vector<KernelConfiguration> configurations;
    size_t index;
};

} // namespace ktt
