#include "catch.hpp"
#include "api/device_info.h"
#include "kernel/kernel_manager.h"

TEST_CASE("Kernel handling operations", "Component: KernelManager")
{
    ktt::DeviceInfo deviceInfo(0, "Device");
    ktt::KernelManager manager(deviceInfo);
    ktt::KernelId id = manager.addKernelFromFile("../tests/test_kernel.cl", "testKernel", ktt::DimensionVector(1024), ktt::DimensionVector(16, 16));

    SECTION("Kernel id is assigned correctly")
    {
        ktt::KernelId secondId = manager.addKernelFromFile("../tests/test_kernel.cl", "testKernel", ktt::DimensionVector(1024),
            ktt::DimensionVector(16, 16));

        REQUIRE(secondId == 1);
    }

    SECTION("Kernel source is loaded correctly")
    {
        std::string source = manager.getKernel(id).getSource();
        std::string expectedSource(std::string("")
            + "__kernel void testKernel(float number, __global float* a, __global float* b, __global float* result)\n"
            + "{\n"
            + "    int index = get_global_id(0);\n"
            + "\n"
            + "    result[index] = a[index] + b[index] + number;\n"
            + "}\n");

        REQUIRE(source == expectedSource);
    }

    SECTION("Parameter with same name cannot be added twice")
    {
        manager.addParameter(id, "param", std::vector<size_t>{1, 2, 3}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
            ktt::Dimension::X);
        REQUIRE_THROWS_AS(manager.addParameter(id, "param", std::vector<size_t>{3}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
            ktt::Dimension::X), std::runtime_error);
    }
}

TEST_CASE("Kernel configuration retrieval", "Component: KernelManager")
{
    ktt::DeviceInfo deviceInfo(0, "Device");
    deviceInfo.setMaxWorkGroupSize(1024);
    ktt::KernelManager manager(deviceInfo);
    ktt::KernelId id = manager.addKernelFromFile("../tests/test_kernel.cl", "testKernel", ktt::DimensionVector(1024), ktt::DimensionVector(16, 16));
    manager.addParameter(id, "param_one", std::vector<size_t>{1, 2, 3}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
        ktt::Dimension::X);
    manager.addParameter(id, "param_two", std::vector<size_t>{5, 10}, ktt::ThreadModifierType::None, ktt::ThreadModifierAction::Add,
        ktt::Dimension::X);

    SECTION("Kernel source with defines is returned correctly")
    {
        std::vector<ktt::ParameterPair> parameterPairs;
        parameterPairs.push_back(ktt::ParameterPair("param_two", static_cast<size_t>(5)));
        parameterPairs.push_back(ktt::ParameterPair("param_one", static_cast<size_t>(2)));

        ktt::KernelConfiguration config(manager.getKernel(id).getGlobalSize(), manager.getKernel(id).getLocalSize(), parameterPairs);
        std::string source = manager.getKernelSourceWithDefines(id, config);
        std::string expectedSource("#define param_one 2\n#define param_two 5\n" + manager.getKernel(id).getSource());

        REQUIRE(source == expectedSource);
    }

    SECTION("Kernel configurations are computed correctly")
    {
        std::vector<ktt::KernelConfiguration> configurations = manager.getKernelConfigurations(id);
        REQUIRE(configurations.size() == 6);
    }
}
