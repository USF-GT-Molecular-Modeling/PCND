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

/*! \file BondedGroupData.cuh
    \brief Defines the helper functions (GPU version) for updating the GPU bonded group tables
 */

#include "util/mgpucontext.h"

#ifndef __BONDED_GROUP_DATA_CUH__
#define __BONDED_GROUP_DATA_CUH__

#ifdef NVCC
//! Sentinel value
const unsigned int GROUP_NOT_LOCAL = 0xffffffff;

//! Storage for group members (GPU declaration)
template<unsigned int group_size>
union group_storage
    {
    unsigned int tag[group_size]; // access 'tags'
    unsigned int idx[group_size]; // access 'indices'
    };

//! Packed group entry for communication (GPU declaration)
template<unsigned int group_size>
struct packed_storage
    {
    group_storage<group_size> tags;  //!< Member tags
    unsigned int type;               //!< Type of bonded group
    unsigned int group_tag;          //!< Tag of this group
    group_storage<group_size> ranks; //!< Current list of member ranks
    };
#else
//! Forward declaration of group_storage
template<unsigned int group_size>
union group_storage;

//! Forward declaration of packed_storage
template<unsigned int group_size>
struct packed_storage;
#endif

template<unsigned int group_size, typename group_t>
void gpu_update_group_table(
    const unsigned int n_groups,
    const unsigned int N,
    const group_t* d_group_table,
    const unsigned int *d_group_type,
    const unsigned int *d_rtag,
    unsigned int *d_n_groups,
    unsigned int max_n_groups,
    unsigned int *d_condition,
    unsigned int next_flag,
    unsigned int &flag,
    group_t *d_pidx_group_table,
    unsigned int *d_pidx_gpos_table,
    const unsigned int pidx_group_table_pitch,
    unsigned int *d_scratch_g,
    unsigned int *d_scratch_idx,
    unsigned int *d_offsets,
    unsigned int *d_seg_offsets,
    mgpu::ContextPtr mgpu_context
    );
#endif // __BONDED_GROUP_DATA_CUH__
