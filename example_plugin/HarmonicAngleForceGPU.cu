// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

#include "HarmonicAngleForceGPU.cuh"
#include "hoomd/TextureTools.h"
#include <stdlib.h>
#include <assert.h>
#include "hoomd/Saru.h"
#include <stdio.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file HarmonicAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic angle forces. Used by HarmonicAngleForceComputeGPU.
*/

//! Texture for reading angle parameters
scalar2_tex_t angle_params_tex;

//! Kernel for caculating harmonic angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_params Parameters for the angle force
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
*/

__global__ void setup_kernel(curandState *state,unsigned int timestep,const unsigned int N, unsigned int time){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}



extern "C" __global__ void gpu_compute_harmonic_angle_forces_kernel(Scalar4* d_force,
                                                                    Scalar* d_virial,
                                                                    const unsigned int virial_pitch,
                                                                    const unsigned int N,
                                                                    const Scalar4 *d_pos,
                                                                    const Scalar2 *d_params,
                                                                    BoxDim box,
                                                                    const group_storage<3> *alist,
                                                                    const unsigned int *apos_list,
                                                                    const unsigned int pitch,
                                                                    const unsigned int *n_angles_list,
                                                                    unsigned int timestep)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (idx >= N)
        return;
    //curandState localState = state[idx];
    curandState localState;
    curand_init(timestep*N*6*idx, 0, 0, &localState);
    //printf("\ntimestep = %i   idx = %i   state = %i\n", timestep,idx,&state);
			
    
		
    

		/*for (int testingindex = 0;testingindex<6;testingindex++)
		{
			Scalar Rtest = curand_uniform(my_curandstate+timestep*N*10+idx*60);
		
			printf("\nTimestep = %i idx = %i Rtest = %f",timestep,idx,Rtest);
			//Scalar Rall[6];
			Rall[testingindex] = Rtest;
		}*/

	
		
		// load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
		int n_angles = n_angles_list[idx];

		// read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
		//Scalar4 idx_postype = d_pos[idx];  // we can be either a, b, or c in the a-b-c triplet
		//Scalar3 idx_pos = make_scalar3(idx_postype.x, idx_postype.y, idx_postype.z);
		//Scalar3 a_pos,b_pos,c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet

		// initialize the force to 0
		Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));

		//Scalar fab[3], fcb[3];

		// loop over all angles
		for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
			{
			float R1 = curand_uniform(&localState);
			float R2 = curand_uniform(&localState);
			float R3 = curand_uniform(&localState);
			float R4 = curand_uniform(&localState);
			float R5 = curand_uniform(&localState);
			float R6 = curand_uniform(&localState);
			//state[idx] = localState;
			//if (idx ==0 || idx ==1)
			//printf("\ntimestep = %i    R1 = %f  R6 = %f\n",timestep,R1,R6);
			
			//printf("\ntimestep = %i  idx = %i   state = %i    R1 = %f   R6 = %f\n",timestep,idx,&state,R1,R6);
			
			
			
			group_storage<3> cur_angle = alist[pitch*angle_idx + idx];

			//int cur_angle_x_idx = cur_angle.idx[0];
			//int cur_angle_y_idx = cur_angle.idx[1];
			int cur_angle_type = cur_angle.idx[2];

			int cur_angle_abc = apos_list[pitch*angle_idx + idx];

			
			// get the angle parameters (MEM TRANSFER: 8 bytes)
			Scalar2 params = texFetchScalar2(d_params, angle_params_tex, cur_angle_type);
			Scalar Xi = params.x;//K
			Scalar tau = params.y;//t_0
			/*
			//get random numbers
			Scalar R1=Rall[0];
			Scalar R2=Rall[1];
			Scalar R3=Rall[2];
			Scalar R4=Rall[3];
			Scalar R5=Rall[4];
			Scalar R6=Rall[5];
			*/
			
			if (cur_angle_abc == 1 && timestep == 0)
			{
				force_idx.x =Xi*sqrt(-2*log(R1))*cosf(2*3.1415926535897*R2);
				force_idx.y =Xi*sqrt(-2*log(R3))*cosf(2*3.1415926535897*R4);
				force_idx.z =Xi*sqrt(-2*log(R5))*cosf(2*3.1415926535897*R6);
				//printf("\n\nSTARING\nidx.x = %f\nidx = %i\n\n",force_idx.x,idx);
				force_idx.w = sqrt(force_idx.x*force_idx.x+force_idx.y*force_idx.y+force_idx.z+force_idx.z);
				d_force[idx] = force_idx;
				//printf("\n\ntimestep = %i\nforce = %f\n",timestep,force_idx.x);
				//printf("\n\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %f\nR2 = %f\nR3 = %f\nR4 = %f\nR5 = %f\nR6 = %f\nforce=%f",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6,force_idx.x);
				//printf("\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %i\nR2 = %i\nR3 = %i\nR4 = %i\nR5 = %i\nR6 = %i\n",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6);
			}
			else if (cur_angle_abc == 1 && timestep != 0)
			{
				Scalar magx=d_force[idx].x;
				Scalar magy=d_force[idx].y;
				Scalar magz=d_force[idx].z;
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
				force_idx.x =E*magx+hx;
				force_idx.y =E*magy+hy;
				force_idx.z =E*magz+hz;
				force_idx.w = sqrt(force_idx.x*force_idx.x+force_idx.y*force_idx.y+force_idx.z+force_idx.z);
				//printf("Xi = %f\n\n",idx.x);
				//printf("\n\ntimestep = %i\nXi = %f\nTau = %f\nR1 = %f\nR2 = %f\nmag = %f\nE = %f\nh = %f\nidx.x= %f\nidx.w = %f\nidx = %i\nangle_idx = %i",timestep,Xi,tau,R1,R2,mag,E,h,force_idx.x,force_idx.w,idx,angle_idx);
				//printf("\ntimestep = %i\nangle_idx = %i\nidx = %i\nR1 = %f\nR2 = %f\nR3 = %f\nR4 = %f\nR5 = %f\nR6 = %f\n",timestep,angle_idx,idx,R1,R2,R3,R4,R5,R6);
				//printf("nangles = %i\npitch = %f\n currangle_x_idx = %i\ncurrangle_y_idx = %i\ncurrangle_type = %i\n",n_angles,pitch,cur_angle_x_idx,cur_angle_y_idx,cur_angle_type);
				d_force[idx] = force_idx;
				//if (timestep%10==0)
				//printf("\n\ntimestep = %i\nforce = %f\n",timestep,force_idx.x);
				//printf("\n\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %f\nR2 = %f\nR3 = %f\nR4 = %f\nR5 = %f\nR6 = %f\nforce=%f",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6,force_idx.x);
				//printf("\ntimestep = %i\ncurangle_abc = %i\nidx = %i\nxidx = %i\nyidx = %i\nR1 = %i\nR2 = %i\nR3 = %i\nR4 = %i\nR5 = %i\nR6 = %i\n",timestep,cur_angle_abc,idx,cur_angle_x_idx,cur_angle_y_idx,R1,R2,R3,R4,R5,R6);
			}
		}
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial arary
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params K and t_0 params packed as Scalar2 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns cudaSuccess in release builds to avoid the cudaThreadSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains K the spring constant
    and the y component contains t_0 the equilibrium angle.
*/
cudaError_t gpu_compute_harmonic_angle_forces(Scalar4* d_force,
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
                                              unsigned int n_angle_types,
                                              int block_size,
                                              const unsigned int compute_capability,
                                              unsigned int timestep)
    {
    assert(d_params);

    static unsigned int max_block_size = UINT_MAX;
    if (max_block_size == UINT_MAX)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, (const void *)gpu_compute_harmonic_angle_forces_kernel);
        max_block_size = attr.maxThreadsPerBlock;
        }

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid( N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind the texture on pre sm 35 arches
    if (compute_capability < 350)
        {
        cudaError_t error = cudaBindTexture(0, angle_params_tex, d_params, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;
        }
	//printf("\n\nTimestep = %i",timestep);
	//Scalar R1 = rand();
	//curandState *devStates;
	//cudaMalloc((void **)&devStates, N*2*sizeof(curandStateMRG32k3a));
	//curandState *d_state;
	//cudaMalloc(&d_state, sizeof(curandState));
	//if (timestep == 1)
	
	//setup_kernel<<<grid, threads>>>(devStates,timestep,N,clock());
	//printf("\n\ndstate = %p\n",d_state);
	
    // run the kernel
    gpu_compute_harmonic_angle_forces_kernel<<< grid, threads>>>(d_force, d_virial, virial_pitch, N, d_pos, d_params, box,
        atable, apos_list, pitch, n_angles_list,timestep);
    

    return cudaSuccess;
    }
