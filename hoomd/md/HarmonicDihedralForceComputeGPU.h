// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard
#include "HarmonicDihedralForceCompute.h"
#include "HarmonicDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

/*! \file HarmonicDihedralForceComputeGPU.h
    \brief Declares the HarmonicDihedralForceGPU class
*/

#ifndef __HARMONICDIHEDRALFORCECOMPUTEGPU_H__
#define __HARMONICDIHEDRALFORCECOMPUTEGPU_H__

//! Implements the harmonic dihedral force calculation on the GPU
/*! HarmonicDihedralForceComputeGPU implements the same calculations as HarmonicDihedralForceCompute,
    but executing on the GPU.

    Per-type parameters are stored in a simple global memory area pointed to by
    \a m_gpu_params. They are stored as Scalar2's with the \a x component being K and the
    \a y component being t_0.

    The GPU kernel can be found in dihedralforce_kernel.cu.

    \ingroup computes
*/
class HarmonicDihedralForceComputeGPU : public HarmonicDihedralForceCompute
    {
    public:
        //! Constructs the compute
        HarmonicDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> system);
        //! Destructor
        ~HarmonicDihedralForceComputeGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            HarmonicDihedralForceCompute::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, int sign, unsigned int multiplicity);

    protected:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
        GPUArray<Scalar4> m_params;           //!< Parameters stored on the GPU (k,sign,m)

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Export the DihedralForceComputeGPU class to python
void export_HarmonicDihedralForceComputeGPU(pybind11::module& m);

#endif