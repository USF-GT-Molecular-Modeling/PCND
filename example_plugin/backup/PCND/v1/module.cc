// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "HarmonicAngleForceComputePCND.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

// specify the python module. Note that the name must expliclty match the PROJECT() name provided in CMakeLists
// (with an underscore in front)
PYBIND11_PLUGIN(_PCND)
    {
    pybind11::module m("_PCND");
    export_HarmonicAngleForceComputePCND(m);

    #ifdef ENABLE_CUDA
    #endif

    return m.ptr();
    }
