/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include "PotentialExternal.h"
#include "PotentialExternalGPU.cuh"
#include "Autotuner.h"

/*! \file PotentialExternalGPU.h
    \brief Declares a class for computing an external potential field on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __POTENTIAL_EXTERNAL_GPU_H__
#define __POTENTIAL_EXTERNAL_GPU_H__

//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
*/
template<class evaluator, cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                                               const typename evaluator::param_type *d_params)>
class PotentialExternalGPU : public PotentialExternal<evaluator>
    {
    public:
        //! Constructs the compute
        PotentialExternalGPU(boost::shared_ptr<SystemDefinition> sysdef,
                             const std::string& log_suffix="");

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            PotentialExternal<evaluator>::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

    protected:

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        boost::scoped_ptr<Autotuner> m_tuner; //!< Autotuner for block size
    };

/*! Constructor
    \param sysdef system definition
 */
template<class evaluator, cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                                               const typename evaluator::param_type *d_params)>
PotentialExternalGPU<evaluator, gpu_cpef>::PotentialExternalGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                                const std::string& log_suffix)
    : PotentialExternal<evaluator>(sysdef, log_suffix)
    {
    this->m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "external_" + evaluator::getName(), this->m_exec_conf));
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator, cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                                               const typename evaluator::param_type *d_params)>
void PotentialExternalGPU<evaluator, gpu_cpef>::computeForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, "PotentialExternalGPU");

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    const BoxDim& box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    this->m_tuner->begin();
    gpu_cpef(external_potential_args_t(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         d_pos.data,
                         box,
                         this->m_tuner->getParam()), d_params.data);
    this->m_tuner->end();

    if (this->m_prof) this->m_prof->pop();

    }

//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialExternalGPU class template.
*/
template < class T, class base >
void export_PotentialExternalGPU(const std::string& name)
    {
    boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<base>, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, const std::string&  >())
                  .def("setParams", &T::setParams)
                  ;
    }

#endif
