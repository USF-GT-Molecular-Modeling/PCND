#include "hoomd/ForceCompute.h"
#include "hoomd/BondedGroupData.h"

#include <memory>

#include <vector>

/*! \file PCNDForceCompute.h
    \brief Declares a class for computing harmonic angles
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __PCNDForceCompute_H__
#define __PCNDForceCompute_H__

//! Computes harmonic angle forces on each particle
/*! Harmonic angle forces are computed on every particle in the simulation.

    The angles which forces are computed on are accessed from ParticleData::getAngleData
    \ingroup computes
*/

class PCNDForceCompute : public ForceCompute
    {
    public:
        //! Constructs the compute
        PCNDForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~PCNDForceCompute();

        //! Set the parameters
        virtual void setParams(unsigned int type, Scalar K, Scalar t_0);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        #ifdef ENABLE_MPI
        //! Get ghost particle fields requested by this pair potential
        /*! \param timestep Current time step
        */
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
                CommFlags flags = CommFlags(0);
                flags[comm_flag::tag] = 1;
                flags |= ForceCompute::getRequestedCommFlags(timestep);
                return flags;
            }
        #endif

    protected:
        Scalar* m_K;    //!< K parameter for multiple angle tyes
        Scalar* m_t_0;  //!< r_0 parameter for multiple angle types

        std::shared_ptr<AngleData> m_angle_data;  //!< Angle data to use in computing angles

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

//! Exports the AngleForceCompute class to python
void export_PCNDForceCompute(pybind11::module& m);

#endif
