#include "PCNDForceCompute.h"

namespace py = pybind11;

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>
#include <boost/random.hpp>
#include <vector>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file PCNDForceCompute.cc
    \brief Contains code for the PCNDForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
PCNDForceCompute::PCNDForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    :  ForceCompute(sysdef), m_Xi(NULL), m_Tau(NULL)
    {

    // access the angle data for later use
    //m_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    //if (m_angle_data->getNTypes() == 0)
        //{
        //m_exec_conf->msg->error() << "angle.harmonic: No angle types specified" << endl;
        //throw runtime_error("Error initializing PCNDForceCompute");
        //}

    // allocate the parameters
    //m_ChainNum = new int[m_angle_data->getNTypes()];
    m_Xi = new Scalar;//[m_angle_data->getNTypes()];
    m_Tau = new Scalar;//[m_angle_data->getNTypes()];
    }

PCNDForceCompute::~PCNDForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PCNDForceCompute" << endl;

    //delete[] m_ChainNum;
    delete[] m_Xi;
    delete[] m_Tau;
    //m_ChainNum = NULL;
    m_Xi = NULL;
    m_Tau = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle in radians for the force computation

    Sets parameters for the potential of a particular angle type
*/
void PCNDForceCompute::setParams(unsigned int type, Scalar Xi, Scalar Tau)
    {
    //m_Xi[type] = Xi;
    //m_Tau[type] = Tau;

    // check for some silly errors a user could make
    if (Xi <= 0)
        m_exec_conf->msg->warning() << "PCND: specified Xi <= 0" << endl;
    if (Tau <= 0)
        m_exec_conf->msg->warning() << "PCND: specified Tau <= 0" << endl;
        
    printf("\n\n\n\n\n\n\n\n\n SET PARAMS \n\n\n\n\n\n\n\n");
    
    //Trace out Bonds to itentify and orient chains///////////////////////////////////////
    
    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);


    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);

    // Zero data for force calculation.
    //memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    //memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    // access the table data
    //ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
    //ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);
	
    // for each of the bonds
    //printf("working sofar \n");
    m_bond_data = m_sysdef->getBondData();
    const unsigned int size = (unsigned int)m_bond_data->getN();
    
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const BondData::members_t bond = m_bond_data->getMembersByIndex(i);
        assert(bond.tag[0] < m_pdata->getN());
        assert(bond.tag[1] < m_pdata->getN());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.tag[0]];
        unsigned int idx_b = h_rtag.data[bond.tag[1]];
        assert(idx_a <= m_pdata->getMaximumTag());
        assert(idx_b <= m_pdata->getMaximumTag());

        // throw an error if this bond is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error() << "bond.table: bond " <<
                bond.tag[0] << " " << bond.tag[1] << " incomplete." << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }
        
        printf("idxa  = %i   idxb = %i  \n",idx_a,idx_b);
		}
    
    }

/*! AngleForceCompute provides
    - \c angle_harmonic_energy
*/
std::vector< std::string > PCNDForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("angle_harmonic_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar PCNDForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("angle_harmonic_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "angle.harmonic: " << quantity << " is not a valid log quantity for AngleForceCompute" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
float rn(void)
   {
   static boost::mt19937 rng(time(NULL));
   static boost::uniform_01<boost::mt19937> zeroone(rng);
   return zeroone();
   }
 
void PCNDForceCompute::computeForces(unsigned int timestep)
    {
		printf("\n\n\ntimestep = %i\n",timestep);
    if (m_prof) m_prof->push("Harmonic Angle");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();
	
	//generate random numbers for PCND's use
	//const unsigned int size = (unsigned int)m_angle_data->getN();
	const unsigned int sizePCND = 50;
	double randomnums[sizePCND*2+6];
	if(timestep == 1)
		{
		for (unsigned int i = 0; i<sizePCND*2+2; i++)
			{
			randomnums[i]=rn();
			//printf("\nrandom number = %f\n", randomnums[i]);
			}
		}
	else
		{
		//printf("\n timestep %i size %i\n", timestep, size);
		for (unsigned int i = 0; i<sizePCND*2+2; i++)
			{
			randomnums[i]=rn();
			//printf("\n i = %i  random number = %f\n", i, randomnums[i]);
			}
		}
		
	
	
	const unsigned int size = (unsigned int)m_bond_data->getN();
	//unsigned int **p = new unsigned int[size][2];
	//unsigned int Bonds [size][2] ={};
    vector<unsigned int> Bonds_A(size);
    vector<unsigned int> Bonds_B(size);
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const BondData::members_t bond = m_bond_data->getMembersByIndex(i);
        assert(bond.tag[0] < m_pdata->getN());
        assert(bond.tag[1] < m_pdata->getN());

        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = h_rtag.data[bond.tag[0]];
        unsigned int idx_b = h_rtag.data[bond.tag[1]];
        assert(idx_a <= m_pdata->getMaximumTag());
        assert(idx_b <= m_pdata->getMaximumTag());

        // throw an error if this bond is incomplete
        if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error() << "PCND Bond Trace " <<
                bond.tag[0] << " " << bond.tag[1] << " incomplete." << endl << endl;
            throw std::runtime_error("Error in PCND Bond Trace");
            }
        
        printf("idxa  = %i   idxb = %i  \n",idx_a,idx_b);
        
        
        Bonds_A[i]=bond.tag[0];
        Bonds_B[i]=bond.tag[1];
        //Bonds[i][0]=bond.tag[0];
        //Bonds[i][1]=bond.tag[1];
        
        printf("Bonds_A  = %i   Bonds_B = %i  \n",Bonds_A[i],Bonds_B[i]);
        
		}
		
		//unsigned int Chains
		//vector<unsigned int> Chains;
		vector< vector<unsigned int> > Chains;
		Chains.resize(1);
		Chains[0].resize(2);
		Chains[0][0]=Bonds_A[0];
		Chains[0][1]=Bonds_B[0];
		//unsigned int Term1=
		printf("Chains = %i   %i\n",Chains[0][0], Chains[0][1]);
		//printf("BondsA begin = %i\n",*Bonds_A.begin());
		//find index////////////////////////////////////////////////
		unsigned int ind = 1;
		unsigned int flag = 0;
		/*
		while (flag == )
		{
			if (Bonds_A[ind]==Chains[0][0])
			{
				flag = 1;
				Chains
			}
		}
		*/
				
		//unsigned int ind = find(*Bonds_A.begin(),*Bonds_A.end(),Chains[0][0]);
		printf("found %i\n",ind);
		
		
    // for each of the angles
/*
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const AngleData::members_t& angle = m_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());
        assert(angle.tag[2] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.tag[0]];
        unsigned int idx_b = h_rtag.data[angle.tag[1]];
        unsigned int idx_c = h_rtag.data[angle.tag[2]];
        
        //printf("testing123\n");
        //printf("\n new angle\n");
        //printf("timestep = %i\n", timestep);
        //printf("particle number 1 is angle.tag %i \n", angle.tag[0]);
        //printf("particle number 1 is idx %i \n", idx_a);
        //printf("particle number 2 is angle.tag %i \n", angle.tag[1]);
        //printf("particle number 2 is idx %i \n", idx_b);
        //printf("particle number 3 is angle.tag %i \n", angle.tag[2]);
        //printf("particle number 3 is idx %i \n", idx_c);
        //int ChainNum=m_ChainNum;
        //float Xi = m_Xi;
        //float Tau = m_Tau;
        //printf("chainnum = %i   Xi = %f   Tau = %f", *m_ChainNum, *m_Xi, *m_Tau);
		
        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL|| idx_b == NOT_LOCAL || idx_c == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error() << "angle.harmonic: angle " <<
                angle.tag[0] << " " << angle.tag[1] << " " << angle.tag[2] << " incomplete." << endl << endl;
            throw std::runtime_error("Error in angle calculation");
            }

        assert(idx_a < m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN()+m_pdata->getNGhosts());

        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x; // used for the 1-3 JL interaction
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars


        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x*dab.x+dab.y*dab.y+dab.z*dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dcb.x*dcb.x+dcb.y*dcb.y+dcb.z*dcb.z;
        Scalar rcb = sqrt(rsqcb);

        Scalar c_abbc = dab.x*dcb.x+dab.y*dcb.y+dab.z*dcb.z;
        c_abbc /= rab*rcb;

        if (c_abbc > 1.0) c_abbc = 1.0;
        if (c_abbc < -1.0) c_abbc = -1.0;

        Scalar s_abbc = sqrt(1.0 - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0/s_abbc;

        // actually calculate the force
        unsigned int angle_type = m_angle_data->getTypeByIndex(i);
        Scalar dth = acos(c_abbc) - m_Tau[angle_type];
        Scalar tk = m_Xi[angle_type]*dth;

        Scalar a = -1.0 * tk * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;

        Scalar fab[3], fcb[3];

        fab[0] = a11*dab.x + a12*dcb.x;
        fab[1] = a11*dab.y + a12*dcb.y;
        fab[2] = a11*dab.z + a12*dcb.z;

        fcb[0] = a22*dcb.x + a12*dab.x;
        fcb[1] = a22*dcb.y + a12*dab.y;
        fcb[2] = a22*dcb.z + a12*dab.z;
        
        
        fab[0] = 0;
        fab[1] = 0;
        fab[2] = 0;
        
        fcb[0] = 0;
        fcb[1] = 0;
        fcb[2] = 0;
        
		//printf("testing123\n");
		
        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = (tk*dth)*Scalar(1.0/6.0);

        // compute 1/3 of the virial, 1/3 for each atom in the angle
        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1./3.) * ( dab.x*fab[0] + dcb.x*fcb[0] );
        angle_virial[1] = Scalar(1./3.) * ( dab.y*fab[0] + dcb.y*fcb[0] );
        angle_virial[2] = Scalar(1./3.) * ( dab.z*fab[0] + dcb.z*fcb[0] );
        angle_virial[3] = Scalar(1./3.) * ( dab.y*fab[1] + dcb.y*fcb[1] );
        angle_virial[4] = Scalar(1./3.) * ( dab.z*fab[1] + dcb.z*fcb[1] );
        angle_virial[5] = Scalar(1./3.) * ( dab.z*fab[2] + dcb.z*fcb[2] );

        // Now, apply the force to each individual atom a,b,c, and accumlate the energy/virial
        // do not update ghost particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += fab[0];
            h_force.data[idx_a].y += fab[1];
            h_force.data[idx_a].z += fab[2];
            h_force.data[idx_a].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*virial_pitch+idx_a]  += angle_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= fab[0] + fcb[0];
            h_force.data[idx_b].y -= fab[1] + fcb[1];
            h_force.data[idx_b].z -= fab[2] + fcb[2];
            h_force.data[idx_b].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*virial_pitch+idx_b]  += angle_virial[j];
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += fcb[0];
            h_force.data[idx_c].y += fcb[1];
            h_force.data[idx_c].z += fcb[2];
            h_force.data[idx_c].w += angle_eng;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*virial_pitch+idx_c]  += angle_virial[j];
            }
        }
*/
    if (m_prof) m_prof->pop();
    }

void export_PCNDForceCompute(py::module& m)
    {
    py::class_<PCNDForceCompute, std::shared_ptr<PCNDForceCompute> >(m, "PCNDForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    .def("setParams", &PCNDForceCompute::setParams)
    ;
    }