KTT - Kernel Tuning Toolkit
===========================

KTT is a C++ tuning library for OpenCL and CUDA kernels. Support for Vulkan compute shaders is work in progress.
Project is currently in late beta stage with all of the baseline functionality available.

Main features
-------------
* Ability to define kernel tuning parameters like thread sizes, vector data types and loop unroll factors in order to fit computation
for particular device
* High-level API which is easier to use than low-level APIs like OpenCL or CUDA driver API
* Shared C++ code between multiple compute APIs, switching between CUDA and OpenCL involves only minor changes in C++ code
(eg. changing path to kernel file)
* Ability to optionally gain greater control over computation by using tuning manipulator, which allows, for example to utilize iterative
kernel launches or run some part of a computation directly in C++ code
* Ability to automatically ensure correctness of tuned computation by providing reference kernel or C++ method
* Large number of utility methods and customization options, for example ability to specify custom tolerance threshold for floating-point
argument validation, ability to specify output target for printing results (C++ stream or file), ability to easily retrieve information about
available devices, etc.
* No direct usage of vendor specific SDKs is needed, only corresponding device drivers have to be installed

Getting started
---------------

* Documentation for KTT API can be found [here](https://github.com/Fillo7/KTT/blob/master/documentation/ktt_api.md).
* Newest version of KTT library can be found [here](https://github.com/Fillo7/KTT/releases).
* Prebuilt binaries are currently available only for some platforms. Other platforms require manual build.

Examples
--------

Examples showcasing KTT functionality are located inside examples folder.
List of currently available examples:

* `compute_api_info (OpenCL / CUDA)`: basic example showing how to retrieve detailed information about compute API platforms and devices through KTT API
* `simple (OpenCL / CUDA)`: basic example showing how to run simple kernel with KTT framework, utilizes reference class, no actual autotuning is done
* `coulomb_sum_2d (OpenCL)`: advanced example which utilizes large number of tuning parameters, thread modifiers and constraints
* `coulomb_sum_3d_iterative (OpenCL)`: 3D version of previous example, utilizes tuning manipulator to iteratively launch 2D kernel
* `coulomb_sum_3d (OpenCL)`: alternative to iterative version, utilizes several tuning parameters and reference kernel
* `nbody (OpenCL)`: advanced example which utilizes tuning parameters, multiple constraints and validation of multiple arguments with reference kernel
* `reduction (OpenCL)`: advanced example which utilizes reference class, tuning manipulator and several tuning parameters

Building KTT
------------

* KTT can be built as a dynamic (shared) library using command line build tool Premake.
Currently supported operating systems are Linux and Windows.

* The prerequisites to build KTT are:
    - C++14 compiler, for example Clang 3.4, GCC 5.0, MSVC 19.0 (Visual Studio 2015) or newer
    - OpenCL or CUDA library, supported SDKs are AMD APP SDK, Intel SDK for OpenCL and NVIDIA CUDA Toolkit
    - [Premake 5](https://premake.github.io/download.html) (alpha 12 or newer)
    
* Build under Linux (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables
    - run `premake5 gmake` to generate makefile
    - run `cd build` to get inside build directory
    - afterwards run `make config={configuration}_{architecture}` to build the project (eg. `make config=release_x64`)
    
* Build under Windows (inside KTT root folder):
    - ensure that path to vendor SDK is correctly set in the environment variables, this should be done automatically during SDK installation
    - run `premake5.exe vs2015` (or `premake5.exe vs2017`) to generate Visual Studio project files
    - open generated solution file and build the project inside Visual Studio

* Following build options are available:
    - `--outdir=path` specifies custom build directory, default build directory is `build`
    - `--platform=vendor` specifies SDK used for building KTT, useful when multiple SDKs are installed
    - `--vulkan` enables inclusion of Vulkan API during compilation (experimental)
    - `--no-examples` disables compilation of examples
    - `--tests` enables compilation of unit tests
    - `--no-cuda` disables inclusion of CUDA API during compilation, only affects Nvidia platform
    
Original project
----------------

KTT is based on [CLTune project](https://github.com/CNugteren/CLTune). Basic KTT API functionality is similar to that of CLTune,
however internal structure was almost completely rewritten from scratch. Portions of code for following features were ported from CLTune:
* Searchers (mainly PSO and annealing searcher)
* Generating of kernel configurations
* Tuning parameter constraints
