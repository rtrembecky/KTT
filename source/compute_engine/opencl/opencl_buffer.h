#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "CL/cl.h"
#include "opencl_command_queue.h"
#include "opencl_utility.h"
#include "kernel_argument/kernel_argument.h"
#include "enum/argument_access_type.h"
#include "enum/argument_data_type.h"
#include "enum/argument_memory_location.h"

namespace ktt
{

class OpenCLBuffer
{
public:
    explicit OpenCLBuffer(const cl_context context, KernelArgument& kernelArgument, const bool zeroCopy) :
        context(context),
        kernelArgumentId(kernelArgument.getId()),
        bufferSize(kernelArgument.getDataSizeInBytes()),
        elementSize(kernelArgument.getElementSizeInBytes()),
        dataType(kernelArgument.getDataType()),
        memoryLocation(kernelArgument.getMemoryLocation()),
        accessType(kernelArgument.getAccessType()),
        openclMemoryFlag(getOpenCLMemoryType(accessType)),
        hostPointer(nullptr),
        zeroCopy(zeroCopy)
    {
        if (memoryLocation == ArgumentMemoryLocation::Host)
        {
            if (!zeroCopy)
            {
                openclMemoryFlag = openclMemoryFlag | CL_MEM_ALLOC_HOST_PTR;
            }
            else
            {
                openclMemoryFlag = openclMemoryFlag | CL_MEM_USE_HOST_PTR;
                hostPointer = kernelArgument.getData();
            }
        }

        cl_int result;
        buffer = clCreateBuffer(context, openclMemoryFlag, bufferSize, hostPointer, &result);
        checkOpenCLError(result, "clCreateBuffer");
    }

    ~OpenCLBuffer()
    {
        checkOpenCLError(clReleaseMemObject(buffer), "clReleaseMemObject");
    }

    void resize(const size_t newBufferSize)
    {
        if (zeroCopy)
        {
            throw std::runtime_error("Cannot resize buffer with CL_MEM_USE_HOST_PTR flag");
        }

        if (bufferSize == newBufferSize)
        {
            return;
        }

        checkOpenCLError(clReleaseMemObject(buffer), "clReleaseMemObject");

        cl_int result;
        buffer = clCreateBuffer(context, openclMemoryFlag, newBufferSize, hostPointer, &result);
        checkOpenCLError(result, "clCreateBuffer");
        bufferSize = newBufferSize;
    }

    void uploadData(cl_command_queue queue, const void* source, const size_t dataSize, cl_event* recordingEvent)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize);
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            if (recordingEvent == nullptr)
            {
                cl_int result = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, dataSize, source, 0, nullptr, nullptr);
                checkOpenCLError(result, "clEnqueueWriteBuffer");
            }
            else
            {
                cl_int result = clEnqueueWriteBuffer(queue, buffer, CL_FALSE, 0, dataSize, source, 0, nullptr, recordingEvent);
                checkOpenCLError(result, "clEnqueueWriteBuffer");
            }
        }
        else
        {
            // Asynchronous buffer operations on mapped memory are currently not supported
            cl_int result;
            void* destination = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_WRITE, 0, dataSize, 0, nullptr, nullptr, &result);
            checkOpenCLError(result, "clEnqueueMapBuffer");

            std::memcpy(destination, source, dataSize);
            checkOpenCLError(clEnqueueUnmapMemObject(queue, buffer, destination, 0, nullptr, recordingEvent), "clEnqueueUnmapMemObject");
        }
    }
    
    void uploadData(cl_command_queue queue, const cl_mem source, const size_t dataSize, cl_event* recordingEvent)
    {
        if (bufferSize < dataSize)
        {
            resize(dataSize);
        }

        if (recordingEvent == nullptr)
        {
            throw std::runtime_error("Recording event for buffer copying operation cannot be null");
        }

        cl_int result = clEnqueueCopyBuffer(queue, source, buffer, 0, 0, dataSize, 0, nullptr, recordingEvent);
        checkOpenCLError(result, "clEnqueueCopyBuffer");
    }

    void downloadData(cl_command_queue queue, void* destination, const size_t dataSize, cl_event* recordingEvent) const
    {
        if (bufferSize < dataSize)
        {
            throw std::runtime_error("Size of data to download is larger than size of buffer");
        }

        if (memoryLocation == ArgumentMemoryLocation::Device)
        {
            if (recordingEvent == nullptr)
            {
                cl_int result = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, dataSize, destination, 0, nullptr, nullptr);
                checkOpenCLError(result, "clEnqueueReadBuffer");
            }
            else
            {
                cl_int result = clEnqueueReadBuffer(queue, buffer, CL_FALSE, 0, dataSize, destination, 0, nullptr, recordingEvent);
                checkOpenCLError(result, "clEnqueueReadBuffer");
            }
        }
        else
        {
            // Asynchronous buffer operations on mapped memory are currently not supported
            cl_int result;
            void* source = clEnqueueMapBuffer(queue, buffer, CL_TRUE, CL_MAP_READ, 0, dataSize, 0, nullptr, nullptr, &result);
            checkOpenCLError(result, "clEnqueueMapBuffer");

            std::memcpy(destination, source, dataSize);
            checkOpenCLError(clEnqueueUnmapMemObject(queue, buffer, source, 0, nullptr, recordingEvent), "clEnqueueUnmapMemObject");
        }
    }

    cl_context getContext() const
    {
        return context;
    }

    ArgumentId getKernelArgumentId() const
    {
        return kernelArgumentId;
    }

    size_t getBufferSize() const
    {
        return bufferSize;
    }

    size_t getElementSize() const
    {
        return elementSize;
    }

    ArgumentDataType getDataType() const
    {
        return dataType;
    }

    ArgumentMemoryLocation getMemoryLocation() const
    {
        return memoryLocation;
    }

    ArgumentAccessType getAccessType() const
    {
        return accessType;
    }

    cl_mem_flags getOpenclMemoryFlag() const
    {
        return openclMemoryFlag;
    }
    
    cl_mem getBuffer() const
    {
        return buffer;
    }

private:
    cl_context context;
    ArgumentId kernelArgumentId;
    size_t bufferSize;
    size_t elementSize;
    ArgumentDataType dataType;
    ArgumentMemoryLocation memoryLocation;
    ArgumentAccessType accessType;
    cl_mem_flags openclMemoryFlag;
    cl_mem buffer;
    void* hostPointer;
    bool zeroCopy;
};

} // namespace ktt
