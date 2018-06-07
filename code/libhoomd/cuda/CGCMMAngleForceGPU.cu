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

#include "CGCMMAngleForceGPU.cuh"
#include "TextureTools.h"

#ifdef WIN32
#include <cassert>
#else
#include <assert.h>
#endif

// small number. cutoff for igoring the angle as being ill defined.
#define SMALL Scalar(0.001)

/*! \file CGCMMAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the CGCMM angle forces. Used by CGCMMAngleForceComputeGPU.
*/

//! Texture for reading angle parameters
scalar2_tex_t angle_params_tex;

//! Texture for reading angle CGCMM S-R parameters
scalar2_tex_t angle_CGCMMsr_tex; // MISSING EPSILON!!! sigma=.x, rcut=.y

//! Texture for reading angle CGCMM Epsilon-pow/pref parameters
scalar4_tex_t angle_CGCMMepow_tex; // now with EPSILON=.x, pow1=.y, pow2=.z, pref=.w

//! Kernel for caculating CGCMM angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
*/
extern "C" __global__ void gpu_compute_CGCMM_angle_forces_kernel(Scalar4* d_force,
                                                                 Scalar* d_virial,
                                                                 const unsigned int virial_pitch,
                                                                 const unsigned int N,
                                                                 const Scalar4 *d_pos,
                                                                 BoxDim box,
                                                                 const group_storage<3> *alist,
                                                                 const unsigned int *apos_list,
                                                                 const unsigned int pitch,
                                                                 const unsigned int *n_angles_list,
                                                                 Scalar2 *d_params,
                                                                 Scalar2 *d_CGCMMsr,
                                                                 Scalar4 *d_CGCMMepow,
                                                                 Scalar* d_random,
                                                                 unsigned int timestep)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles =n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
    Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
    Scalar3 a_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch*angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];

        // store the a and c positions to accumlate their forces
        int cur_angle_type = cur_angle.idx[2];
        int cur_angle_abc = apos_list[pitch*angle_idx + idx];

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        Scalar3 x_pos = make_scalar3(x_postype.x, x_postype.y, x_postype.z);
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        Scalar3 y_pos = make_scalar3(y_postype.x, y_postype.y, y_postype.z);

        if (cur_angle_abc == 1)
            {
            a_pos = x_pos;
            c_pos = y_pos;

            // calculate dr for a-b,c-b,and a-c
            //Scalar3 dab = a_pos - b_pos;
            //Scalar3 dcb = c_pos - b_pos;
            Scalar3 dac = a_pos - c_pos;

            // apply periodic boundary conditions
            //dab = box.minImage(dab);
            //dcb = box.minImage(dcb);
            dac = box.minImage(dac);

            // get the angle parameters (MEM TRANSFER: 8 bytes)
            Scalar2 params = texFetchScalar2(d_params, angle_params_tex, cur_angle_type);
            Scalar omega_tau = params.x;
            //Scalar tau = params.y;

            //Scalar rsqab = dot(dab, dab);
            //Scalar rab = sqrtf(rsqab);
            //Scalar rsqcb = dot(dcb, dcb);;
            //Scalar rcb = sqrtf(rsqcb);
            Scalar rsqac = dot(dac, dac);
            Scalar rac = sqrtf(rsqac);
            dac = dac/rac;
            
            const Scalar2 cgSR = texFetchScalar2(d_CGCMMsr, angle_CGCMMsr_tex, cur_angle_type);
            //Scalar2 cgSR = tex1Dfetch(angle_CGCMMsr_tex, cur_angle_type);
            Scalar E = cgSR.x;
            Scalar chainnumber = cgSR.y;

            //magnitude of  previous force
            Scalar mag=d_force[idx].w;
            if (omega_tau!= 0)
                {
                Scalar h=0;
                int i=chainnumber*2;
                h=sqrtf((-2*omega_tau)*(1-E*E)*logf(d_random[i-2]))*cosf(2*3.1415926535897*d_random[i-1]);
                mag=mag*E+h;
                if (timestep==0)
                    {
                    mag=sqrtf((-2*omega_tau)*logf(d_random[i-2]))*cosf(2*3.1415926535897*d_random[i-1]);
                    }
                Scalar tempmag=0;
                
                tempmag = sqrtf((-2*omega_tau)*logf(0.001));
                if (mag>tempmag)
                    {
                    mag=tempmag;
                    }
                else if (mag<-tempmag)
                    {
                    mag=-tempmag;
                    }

                force_idx.x += dac.x * mag;
                force_idx.y += dac.y * mag;
                force_idx.z += dac.z * mag;
                force_idx.w += mag;
                }
            else
                {
                force_idx.x += 0;
                force_idx.y += 0;
                force_idx.z += 0;
                force_idx.w += 0;
                }
            }

        if (cur_angle_abc == 0 && n_angles==2)
            { 
            force_idx.x += 0;
            force_idx.y += 0;
            force_idx.z += 0;
            force_idx.w += 0;
            }




        if (cur_angle_abc == 2 && n_angles==2)
            { 
            force_idx.x += 0;
            force_idx.y += 0;
            force_idx.z += 0;
            force_idx.w += 0;
            }
        }
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params K and t_0 params packed as Scalar2 variables
    \param d_CGCMMsr sigma, and rcut packed as a Scalar2
    \param d_CGCMMepow epsilon, pow1, pow2, and prefactor packed as a Scalar4
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains K the spring constant
    and the y component contains t_0 the equilibrium angle.
*/
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
                                           Scalar* d_random,
                                           unsigned int timestep)
    {
    assert(d_params);
    assert(d_CGCMMsr);
    assert(d_CGCMMepow);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_CGCMM_angle_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( (int)ceil((double)N / (double)run_block_size), 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind the textures on pre sm 35 arches
    if (compute_capability < 350)
        {
        cudaError_t error = cudaBindTexture(0, angle_params_tex, d_params, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;

        error = cudaBindTexture(0, angle_CGCMMsr_tex, d_CGCMMsr, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;

        error = cudaBindTexture(0, angle_CGCMMepow_tex, d_CGCMMepow, sizeof(Scalar4) * n_angle_types);
        if (error != cudaSuccess)
            return error;
        }

    // run the kernel
    gpu_compute_CGCMM_angle_forces_kernel<<< grid, threads>>>(d_force,
                                                              d_virial,
                                                              virial_pitch,
                                                              N,
                                                              d_pos,
                                                              box,
                                                              atable,
                                                              apos_list,
                                                              pitch,
                                                              n_angles_list,
                                                              d_params,
                                                              d_CGCMMsr,
                                                              d_CGCMMepow,
                                                              d_random,
                                                              timestep);
    return cudaSuccess;
    }
