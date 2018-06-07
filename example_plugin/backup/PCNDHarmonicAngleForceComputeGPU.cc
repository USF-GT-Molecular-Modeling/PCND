// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard

/*! \file PCNDHarmonicAngleForceComputeGPU.cc
    \brief Defines PCNDHarmonicAngleForceComputeGPU
*/



#include "PCNDHarmonicAngleForceComputeGPU.h"
#include "hoomd/Saru.h"
#include <time.h>
#include <cuda.h>
#include <curand.h>


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    ;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    ;}} while(0)

namespace py = pybind11;

using namespace std;

int testint;
curandGenerator_t gen2;

//float *hostData, *devData;
//curandGenerator_t gen;
//int numtest;


/*! \param sysdef System to compute angle forces on
*/
PCNDHarmonicAngleForceComputeGPU::PCNDHarmonicAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
        : PCNDHarmonicAngleForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a AngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing AngleForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar2> params(m_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "harmonic_angle", this->m_exec_conf));

    
    }

PCNDHarmonicAngleForceComputeGPU::~PCNDHarmonicAngleForceComputeGPU()
    {
    }
    
    
    
    
//int PCNDHarmonicAngleForceComputeGPU::SetSeed(int Seed, int num)
//{
	
//}

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void PCNDHarmonicAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0)
    {
    PCNDHarmonicAngleForceCompute::setParams(type, K, t_0);

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(K, t_0);
    
    testint= 46;
    //PCNDHarmonicAngleForceComputeGPU::SetSeed(seed,N);
    
    size_t n2 = 6;
    size_t i;
    //curandGenerator_t gen;
    float *devData2, *hostData2;
    //float *devData;

    /* Allocate n floats on host */
    hostData2 = (float *)calloc(n2, sizeof(float));

    /* Allocate n floats on device */
    cudaMalloc((void **)&devData2, n2*sizeof(float));

    /* Create pseudo-random number generator */
    curandCreateGenerator(&gen2,CURAND_RNG_PSEUDO_DEFAULT);
    
    /* Set seed */
    curandSetPseudoRandomGeneratorSeed(gen2,time(NULL)+clock());
    //printf("\n gen2paramset = %p\n",gen2);
    curandGenerateUniform(gen2, devData2, n2);
    /* Copy device memory to host */
    cudaMemcpy(hostData2, devData2, n2 * sizeof(float),cudaMemcpyDeviceToHost);
    //printf("\n gen2paramset2= %p\n",gen2);
    /* Show result */
    printf("\n\nParamset1");
    for(i = 0; i < 6; i++) {
        printf("%1.4f ", hostData2[i]);
    }
    
    curandGenerateUniform(gen2, devData2, n2);
    /* Copy device memory to host */
    cudaMemcpy(hostData2, devData2, n2 * sizeof(float),cudaMemcpyDeviceToHost);
    //printf("\n gen2paramset2= %p\n",gen2);
    /* Show result */
    printf("\n\nParamset2");
    for(i = 0; i < 6; i++) {
        printf("%1.4f ", hostData2[i]);
    }
    
    
    //return EXIT_SUCCESS;
    
    }
    


/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_angle_forces to do the dirty work.
*/
void PCNDHarmonicAngleForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "Harmonic Angle");

    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);
    //ArrayHandle<Scalar> hostData(devData, access_location::host, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_angle_data->getGPUTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_angle_data->getGPUPosTable(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_angle_data->getNGroupsArray(), access_location::device, access_mode::read);
	const unsigned int N = m_pdata->getN();
	//printf("\n\n\n\ntestint = %i\n\n\n\n\n",testint);
	

	
	//hoomd::detail::Saru rng(time(0),timestep,rand());
	//const unsigned int size = (unsigned int)m_angle_data->getN();
	//GPUArray<Scalar> m_random (size*6,exec_conf);
    //ArrayHandle<Scalar> d_random(m_random,access_location::host,access_mode::overwrite);
    //for (unsigned int i = 0; i < size*6; i++)
	//{
		//printf("\nI = %i     R = %f\n",i,d_random.data[i]);
	//}
    
    /*
	//GPUArray<float> d_random;
	for (unsigned int i = 0; i < size*6; i++)
	{
		d_random.data[i]=rng.s<Scalar>(0,1);
		//printf("\ni = %i     R = %f\n",i,d_random.data[i]);
		
	}
	* */
	//d_random.data[0]=0;
		
		
	//Scalar R1 = rng.s<Scalar>(0,1);
	
	//float *devData;
	
	/* Allocate n floats on device */
	size_t n2 = N*6;
	size_t i;
	float *devData2;
	float *hostData2;//jkh
	/* Allocate n floats on host */
    hostData2 = (float *)calloc(n2, sizeof(float));
    cudaMalloc((void **)&devData2, n2*sizeof(float));
	curandGenerateUniform(gen2, devData2, n2);
    /* Copy device memory to host */
    cudaMemcpy(hostData2, devData2, n2 * sizeof(float),cudaMemcpyDeviceToHost);
    //printf("\n gen2paramset2= %p\n",gen2);
    /* Show result */
    printf("\n\nTimesetp = %i\n",timestep);
    for(i = 0; i < 6; i++) {
        printf("%1.4f ", hostData2[i]);
    }
    //printf("\n gen2cctimestep = %p\n",gen2);

    // run the kernel on the GPU
    m_tuner->begin();
    gpu_compute_harmonic_angle_forces(d_force.data,
                                      d_virial.data,
                                      m_virial.getPitch(),
                                      m_pdata->getN(),
                                      d_pos.data,
                                      box,
                                      d_gpu_anglelist.data,
                                      d_gpu_angle_pos_list.data,
                                      m_angle_data->getGPUTableIndexer().getW(),
                                      d_gpu_n_angles.data,
                                      d_params.data,
                                      m_angle_data->getNTypes(),
                                      m_tuner->getParam(),
                                      m_exec_conf->getComputeCapability(),
                                      timestep,
                                      devData2);
                                     

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_PCNDHarmonicAngleForceComputeGPU(py::module& m)
    {
    py::class_<PCNDHarmonicAngleForceComputeGPU, std::shared_ptr<PCNDHarmonicAngleForceComputeGPU> >(m, "PCNDHarmonicAngleForceComputeGPU", py::base<PCNDHarmonicAngleForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    ;
    }
