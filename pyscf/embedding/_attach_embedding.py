#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Attach Embedding to SCF
'''

from pyscf import lib
from pyscf.lib import logger


def _for_scf(mf, embedding_obj):
    '''Add embedding to SCF (HF and DFT) method.
    '''
    if isinstance(mf, _Embedding):
        mf.with_embedding = embedding_obj
        return mf

    sol_mf = SCFWithEmbedding(mf, embedding_obj)
    name = embedding_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithEmbedding, mf.__class__), name)


class _Embedding:
    pass


class SCFWithEmbedding(_Embedding):
    _keys = {'with_embedding'}

    def __init__(self, mf, embedding_obj):
        self.__dict__.update(mf.__dict__)
        self.with_embedding = embedding_obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_embedding.check_sanity()
        self.with_embedding.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_embedding.reset(mol)
        return super().reset(mol)

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        vhf = super().get_veff(mol, dm, *args, **kwargs)
        with_embedding = self.with_embedding
        with_embedding.e, with_embedding.v = with_embedding.kernel(dm)
        e_embedding, v_embedding = with_embedding.e, with_embedding.v

        # NOTE: v_embedding should not be added to vhf in this place. This is
        # because vhf is used as the reference for direct_scf in the next
        # iteration. If v_embedding is added here, it may break direct SCF.
        return lib.tag_array(vhf, e_embedding=e_embedding, v_embedding=v_embedding)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.info(self, '\n******** %s Energy Contributions ********', self.with_embedding.
                    __class__.__name__)
        logger.info(self, 'Electrostatic Contributions (E_es) = %.15g', self.with_embedding._e_es)
        logger.info(self, 'Induced Contributions (E_ind) = %.15g', self.with_embedding._e_ind)
        if 'vdw' in self.with_embedding.options:
            logger.info(self, 'Repulsion Contributions (E_rep) = %.15g', self.with_embedding._e_rep)
            logger.info(self, 'Dispersion Contributions (E_disp) = %.15g', self.with_embedding._e_disp)
        if self.with_embedding._environment_energy:
            logger.info(self, 'Environment Contributions (E_mul) = %.15g', self.with_embedding.
                        classical_subsystem.environment_energy)
        logger.info(self, '\n')
        if self.converged:
            logger.note(self, 'converged SCF energy = %.15g', self.e_tot)
        else:
            logger.note(self, 'SCF not converged.')
            logger.note(self, 'SCF energy = %.15g', self.e_tot)
        return self

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        # DIIS was called inside super().get_fock. v_embedding, as a function of
        # dm, should be extrapolated as well. To enable it, v_embedding has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf, 'v_embedding', None) is None:
            vhf = self.get_veff(self.mol, dm)
        return super().get_fock(h1e, s1e, vhf + vhf.v_embedding, dm, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf, 'e_embedding', None) is None:
            vhf = self.get_veff(self.mol, dm)
        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_tot += vhf.e_embedding
        self.scf_summary['e_embedding'] = vhf.e_embedding.real
        logger.debug(self, 'Embedding Energy = %.15g', vhf.e_embedding)
        return e_tot, e_coul

    def nuc_grad_method(self):
        grad_method = super().nuc_grad_method()
        return self.with_embedding.nuc_grad_method(grad_method)

    Gradients = nuc_grad_method
