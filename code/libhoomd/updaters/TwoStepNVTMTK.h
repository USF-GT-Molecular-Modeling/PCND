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

// Maintainer: joaander

#include "IntegrationMethodTwoStep.h"
#include "Variant.h"
#include "ComputeThermo.h"

#ifndef __TWO_STEP_NVT_MTK_H__
#define __TWO_STEP_NVT_MTK_H__

/*! \file TwoStepNVTMTK.h
    \brief Declares the TwoStepNVTMTK class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps in the NVT ensemble
/*! Implements Martyna-Tobias-Klein (MTK) NVT integration through the IntegrationMethodTwoStep interface

    Integrator variables mapping:
     - [0] -> xi
     - [1] -> eta

    The instantaneous temperature of the system is computed with the provided ComputeThermo. Correct dynamics require
    that the thermo computes the temperature of the assigned group and with D*N-D degrees of freedom. TwoStepNVTMTK does
    not check for these conditions.

    For the update equations of motion, see Refs. \cite{Martyna1994,Martyna1996}

    \ingroup updaters
*/
class TwoStepNVTMTK : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepNVTMTK(boost::shared_ptr<SystemDefinition> sysdef,
                   boost::shared_ptr<ParticleGroup> group,
                   boost::shared_ptr<ComputeThermo> thermo,
                   Scalar tau,
                   boost::shared_ptr<Variant> T,
                   const std::string& suffix = std::string(""));
        virtual ~TwoStepNVTMTK();

        //! Update the temperature
        /*! \param T New temperature to set
        */
        virtual void setT(boost::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        //! Update the tau value
        /*! \param tau New time constant to set
        */
        virtual void setTau(Scalar tau)
            {
            m_tau = tau;
            }

        //! Set the value of xi (for unit tests)
        void setXi(Scalar new_xi)
            {
            IntegratorVariables v = getIntegratorVariables();
            Scalar& xi = v.variable[0];
            xi = new_xi;
            setIntegratorVariables(v);
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        boost::shared_ptr<ComputeThermo> m_thermo;    //!< compute for thermodynamic quantities

        Scalar m_tau;                   //!< tau value for Nose-Hoover
        boost::shared_ptr<Variant> m_T; //!< Temperature set point
        std::string m_log_name;         //!< Name of the reservior quantity that we log

        Scalar m_exp_thermo_fac;        //!< Thermostat rescaling factor
        Scalar m_curr_T;                //!< Current temperature

        // advance the thermostat
        /*!\param timestep The time step
         * \param broadcast True if we should broadcast the integrator variables via MPI
         */
        void advanceThermostat(unsigned int timestep, bool broadcast=true);
    };

//! Exports the TwoStepNVTMTK class to python
void export_TwoStepNVTMTK();

#endif // #ifndef __TWO_STEP_NVT_MTK_H__
