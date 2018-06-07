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

// Maintainer: dnlebard

/*! \file CGCMMAngleForceComputeGPU.cc
    \brief Defines CGCMMAngleForceComputeGPU
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "CGCMMAngleForceComputeGPU.h"
#include <time.h>
#include <vector>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
//using namespace boost::random;

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute angle forces on
*/
CGCMMAngleForceComputeGPU::CGCMMAngleForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : CGCMMAngleForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a CGCMMAngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing CGCMMAngleForceComputeGPU");
        }


    // allocate and zero device memory
    GPUArray<Scalar2> params (m_CGCMMAngle_data->getNTypes(),exec_conf);
    m_params.swap(params);
    GPUArray<Scalar2> CGCMMsr(m_CGCMMAngle_data->getNTypes(),exec_conf);
    m_CGCMMsr.swap(CGCMMsr);
    GPUArray<Scalar4> CGCMMepow(m_CGCMMAngle_data->getNTypes(),exec_conf);
    m_CGCMMepow.swap(CGCMMepow);

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "cgcmm_angle", this->m_exec_conf));
    }

CGCMMAngleForceComputeGPU::~CGCMMAngleForceComputeGPU()
    {
    }



float rn(void)
    {
    static boost::mt19937 rng(time(NULL));
    static boost::uniform_01<boost::mt19937> zeroone(rng);
    return zeroone();
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation
    \param cg_type the type of course grained angle we are using
    \param eps the well depth
    \param sigma the particle radius

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void CGCMMAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma)
    {
    CGCMMAngleForceCompute::setParams(type, K, t_0, cg_type, eps, sigma);

    const Scalar myPow1 = 1;
    const Scalar myPow2 = 1;
    const Scalar myPref = 1;

    Scalar my_rcut = cg_type;

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_CGCMMsr(m_CGCMMsr, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_CGCMMepow(m_CGCMMepow, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(K, t_0);
    h_CGCMMsr.data[type] = make_scalar2(sigma, my_rcut);
    h_CGCMMepow.data[type] = make_scalar4(eps, myPow1, myPow2, myPref);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_CGCMM_angle_forces to do the dirty work.
*/
void CGCMMAngleForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "CGCMM Angle");

    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    //Not necessary - force and virial are zeroed in the kernel
    //m_force.memclear();
    //m_virial.memclear();
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_CGCMMsr(m_CGCMMsr, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_CGCMMepow(m_CGCMMepow, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> h_CGCMMepow(m_CGCMMepow, access_location::host, access_mode::read);
    

    unsigned int n=h_CGCMMepow.data[0].x*2+6;

    GPUArray<float> m_random (n,exec_conf);
    ArrayHandle<float> h_random(m_random,access_location::host,access_mode::overwrite);
    unsigned int i;

    for (i=0; i<(n); i++)
        {
        h_random.data[i]=rn();
        }

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_CGCMMAngle_data->getGPUTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_CGCMMAngle_data->getGPUPosTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_CGCMMAngle_data->getNGroupsArray(), access_location::device, access_mode::read);

    ArrayHandle<float> d_random(m_random,access_location::device,access_mode::read);

    // run the kernel
    m_tuner->begin();
    gpu_compute_CGCMM_angle_forces(d_force.data,
                                   d_virial.data,
                                   m_virial.getPitch(),
                                   m_pdata->getN(),
                                   d_pos.data,
                                   box,
                                   d_gpu_anglelist.data,
                                   d_gpu_angle_pos_list.data,
                                   m_CGCMMAngle_data->getGPUTableIndexer().getW(),
                                   d_gpu_n_angles.data,
                                   d_params.data,
                                   d_CGCMMsr.data,
                                   d_CGCMMepow.data,
                                   m_CGCMMAngle_data->getNTypes(),
                                   m_tuner->getParam(),
                                   m_exec_conf->getComputeCapability(),
                                   d_random.data,
                                   timestep);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(exec_conf);
    }

void export_CGCMMAngleForceComputeGPU()
    {
    class_<CGCMMAngleForceComputeGPU, boost::shared_ptr<CGCMMAngleForceComputeGPU>, bases<CGCMMAngleForceCompute>, boost::noncopyable >
    ("CGCMMAngleForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }
