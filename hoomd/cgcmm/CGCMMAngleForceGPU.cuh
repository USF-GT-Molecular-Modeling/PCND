// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"

/*! \file CGCMMAngleForceGPU.cuh
    \brief Declares GPU kernel code for calculating the CGCMM angle forces. Used by CGCMMAngleForceComputeGPU.
*/

#ifndef __CGCMMANGLEFORCEGPU_CUH__
#define __CGCMMANGLEFORCEGPU_CUH__

//! Kernel driver that computes harmonic angle forces for HarmonicAngleForceComputeGPU
cudaError_t gpu_compute_CGCMM_angle_forces(Scalar4* d_force,
                                           Scalar* d_virial,
                                           const unsigned int virial_pitch,
                                           const unsigned int N,
                                           const Scalar4 *d_pos,
                                           const BoxDim& box,
                                           const group_storage<3> *atable,
                                           const unsigned int *apos_list,
                                           const unsigned int pitch,
                                           const unsigned int *n_angles_list,
                                           Scalar2 *d_params,
                                           Scalar2 *d_CGCMMsr,
                                           Scalar4 *d_CGCMMepow,
                                           unsigned int n_angle_types,
                                           int block_size,
                                           const unsigned int compute_capability,
                                           unsigned int timestep,
                                           float *hostData,
                                           int PCNDtimestep,
                                           float *hostCarryover);

#endif
