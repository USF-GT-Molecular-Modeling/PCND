// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "CGCMMAngleForceGPU.cuh"
#include "hoomd/TextureTools.h"
#include <cuda.h>
#include <curand_kernel.h>

#include <assert.h>

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
                                                                 unsigned int timestep,
                                                                 float *devData,
                                                                 int PCNDtimestep,
                                                                 float *devCarryover)
    {
		// start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (idx >= N)
        return;
    //curandState localState = state[idx];
    curandState localState;

		
		// load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
		int n_angles = n_angles_list[idx];


		// initialize the force to 0
		Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

		//make counter for number of loops
		int counter = 0;
		
		// loop over all angles
		for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
		{
			int cur_angle_abc = apos_list[pitch*angle_idx + idx];
			if (cur_angle_abc ==1)
			{
				counter += 1;
				////////////////////////////////////////////////////////////Get params
				group_storage<3> cur_angle = alist[pitch*angle_idx + idx];

				//int cur_angle_x_idx = cur_angle.idx[0];
				//int cur_angle_y_idx = cur_angle.idx[1];
				int cur_angle_type = cur_angle.idx[2];

				

				
				// get the angle parameters (MEM TRANSFER: 8 bytes)
				Scalar2 params = texFetchScalar2(d_params, angle_params_tex, cur_angle_type);
				Scalar Xi = params.x;//K
				Scalar tau = params.y;//t_0
				
				////////////////// get sig params
				const Scalar2 cgSR = texFetchScalar2(d_CGCMMsr, angle_CGCMMsr_tex, cur_angle_type);

				int number = cgSR.x;//sigma//number
				//Scalar cgrcut = cgSR.y;
				//////////////////////
				
				/////////////get eps param
				const Scalar4 cgEPOW = texFetchScalar4(d_CGCMMepow, angle_CGCMMepow_tex, cur_angle_type);

				// get the angle pow/pref parameters (MEM TRANSFER: 12 bytes)
				int seed = cgEPOW.x;
				//Scalar cgpow1 = cgEPOW.y;
				//Scalar cgpow2 = cgEPOW.z;
				//Scalar cgpref = cgEPOW.w;
				//////////////////////////////////////////////////////////////
				//curand_init(timestep*N*6*(idx+1)*seed, 0, 0, &localState);
				/*
				float R1 = curand_uniform(&localState);
				float R2 = curand_uniform(&localState);
				float R3 = curand_uniform(&localState);
				float R4 = curand_uniform(&localState);
				float R5 = curand_uniform(&localState);
				float R6 = curand_uniform(&localState);
				*/
				float R1 = devData[(number)*6];
				float R2 = devData[(number)*6+1];
				float R3 = devData[(number)*6+2];
				float R4 = devData[(number)*6+3];
				float R5 = devData[(number)*6+4];
				float R6 = devData[(number)*6+5];
				
				
				//state[idx] = localState;
				//if (timestep == 4 && angle_idx == n_angles-1)
				//printf("\ntimestep = %i  idx = %i  angle_idx = %i  R1 = %f  R6 = %f   number = %i\n",timestep,idx,angle_idx,R1,R6,number);
				
				//printf("\ntimestep = %i  idx = %i   state = %i    R1 = %f   R6 = %f\n",timestep,idx,&state,R1,R6);

				
				if (PCNDtimestep == 0)
				{
					devCarryover[(number)*6+counter*3]=Xi*sqrt(-2*log(R1))*cosf(2*3.1415926535897*R2);
					devCarryover[(number)*6+1+counter*3]=Xi*sqrt(-2*log(R3))*cosf(2*3.1415926535897*R4);
					devCarryover[(number)*6+2+counter*3]=Xi*sqrt(-2*log(R5))*cosf(2*3.1415926535897*R6);
					
					
					force_idx.x +=devCarryover[(number)*6+counter*3];
					force_idx.y +=devCarryover[(number)*6+1+counter*3];
					force_idx.z +=devCarryover[(number)*6+2+counter*3];
					//printf("\n\nSTARING\nidx.x = %f\nidx = %i\n\n",force_idx.x,idx);
					force_idx.w += sqrt(force_idx.x*force_idx.x+force_idx.y*force_idx.y+force_idx.z*force_idx.z);
					
					
					//printf("\n\ntimestep = %i\nforce = %f\n",timestep,force_idx.x);
					//printf("\n\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %f\nR2 = %f\nR3 = %f\nR4 = %f\nR5 = %f\nR6 = %f\nforce=%f",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6,force_idx.x);
					//printf("\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %i\nR2 = %i\nR3 = %i\nR4 = %i\nR5 = %i\nR6 = %i\n",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6);
					
				}
				else if (PCNDtimestep != 0)
				{
					Scalar magx=devCarryover[(number)*6+counter*3];
					Scalar magy=devCarryover[(number)*6+1+counter*3];
					Scalar magz=devCarryover[(number)*6+2+counter*3];
					
					Scalar E = exp(-1/tau);
					Scalar hx = Xi*sqrt(-2*(1-E*E)*log(R1))*cosf(2*3.1415926535897*R2);
					Scalar hy = Xi*sqrt(-2*(1-E*E)*log(R3))*cosf(2*3.1415926535897*R4);
					Scalar hz = Xi*sqrt(-2*(1-E*E)*log(R5))*cosf(2*3.1415926535897*R6);
					if (hx>Xi*sqrt(-2*log(0.001)))
					{
						hx=Xi*sqrt(-2*log(0.001));
					} else if (hx<-Xi*sqrt(-2*log(0.001)))
					{
						hx=-Xi*sqrt(-2*log(0.001));
					}
					if (hy>Xi*sqrt(-2*log(0.001)))
					{
						hy=Xi*sqrt(-2*log(0.001));
					} else if (hy<-Xi*sqrt(-2*log(0.001)))
					{
						hy=-Xi*sqrt(-2*log(0.001));
					}
					if (hz>Xi*sqrt(-2*log(0.001)))
					{
						hz=Xi*sqrt(-2*log(0.001));
					} else if (hz<-Xi*sqrt(-2*log(0.001)))
					{
						hz=-Xi*sqrt(-2*log(0.001));
					}
					
					devCarryover[(number)*6+counter*3]=E*magx+hx;
					devCarryover[(number)*6+1+counter*3]=E*magy+hy;
					devCarryover[(number)*6+2+counter*3]=E*magz+hz;
					
					force_idx.x +=devCarryover[(number)*6+counter*3];
					force_idx.y +=devCarryover[(number)*6+1+counter*3];
					force_idx.z +=devCarryover[(number)*6+2+counter*3];
					//if (idx==0)
					//printf("\ntimestep = %i  idx = %i  magx = %f  R1 = %f  R2 = %f  E = %f  hx = %f  force_x = %f\n",timestep,idx,magx,R1,R2,E,hx,force_idx.x);
					force_idx.w += sqrt(force_idx.x*force_idx.x+force_idx.y*force_idx.y+force_idx.z*force_idx.z);
					//force_idx.w=force_idx.x;
					//if (timestep%100==0)
					//{
					//printf("\ntimestep = %i  idx = %i  angle_idx = %i  R1 = %f  R6 = %f   cur_angle_type = %i   cur_angle_abc = %i   Xi = %f   Tau = %f  seed = %i  number = %f   Fx = %f   Fy = %f   Fz = %f  Fw= %f\n",timestep,idx,angle_idx,R1,R6,cur_angle_type,cur_angle_abc,Xi,tau,seed,number,force_idx.x,force_idx.y,force_idx.z,force_idx.w);
					//}
					//printf("Xi = %f\n\n",idx.x);
					//printf("\n\ntimestep = %i\nXi = %f\nTau = %f\nR1 = %f\nR2 = %f\nmag = %f\nE = %f\nh = %f\nidx.x= %f\nidx.w = %f\nidx = %i\nangle_idx = %i",timestep,Xi,tau,R1,R2,mag,E,h,force_idx.x,force_idx.w,idx,angle_idx);
					//printf("\ntimestep = %i\nangle_idx = %i\nidx = %i\nR1 = %f\nR2 = %f\nR3 = %f\nR4 = %f\nR5 = %f\nR6 = %f\n",timestep,angle_idx,idx,R1,R2,R3,R4,R5,R6);
					//printf("nangles = %i\npitch = %f\n currangle_x_idx = %i\ncurrangle_y_idx = %i\ncurrangle_type = %i\n",n_angles,pitch,cur_angle_x_idx,cur_angle_y_idx,cur_angle_type);
					
					//if (timestep%1==0 && idx==0)
					//{
					//	printf("\n\ntimestep = %i  idx = %i   R1 = %f   R2 = %f   number = %i   Xi = %f   tau = %f  Force = %f\n\n",timestep,idx,R1,R2,number,Xi,tau,force_idx.w);
					//}
					//printf("\n\ntimestep = %i\nforce = %f\n",timestep,force_idx.x);
					//printf("\n\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %f\nR2 = %f\nR3 = %f\nR4 = %f\nR5 = %f\nR6 = %f\nforce=%f",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6,force_idx.x);
					//printf("\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %i\nR2 = %i\nR3 = %i\nR4 = %i\nR5 = %i\nR6 = %i\n",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6);
				}
				d_force[idx] = force_idx;
			}
		}
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
                                           unsigned int timestep,
                                           float *devData,
                                           int PCNDtimestep,
                                           float *devCarryover)
    {
    assert(d_params);
    assert(d_CGCMMsr);
    assert(d_CGCMMepow);
    
    //float *lookupArray;
    //cudaMemcpyToSymbol(lookupArray,(void*)hostData,100*sizeof(float),0);

    if (N == 0)
        return cudaSuccess;

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
                                                              timestep,
                                                              devData,
                                                              PCNDtimestep,
                                                              devCarryover);

    return cudaSuccess;
    }
