#!/usr/bin/env python
# Copyright 2019-2020 The PySCF Developers. All Rights Reserved.
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
Analytical nuclear gradients for Embedding
'''

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df, scf
from pyscf.embedding._attach_embedding import _Embedding
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad

try:
    import pyframe
except ImportError:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.')


def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''

    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Embedding)
    logger.warn(grad_method, "PE gradients are not optimized for performance.")
    if not isinstance(grad_method.base, scf.hf.SCF):
        raise NotImplementedError("PE gradients only implemented for SCF methods.")

    grad_method_class = grad_method.__class__

    class EmbeddingGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        # TODO: if moving to python3, change signature to
        # def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        def kernel(self, *args, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)

            self.de_solvent = kernel(self.base.with_embedding, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_embedding.__class__.__name__)
                rhf_grad._write(self, self.mol, self.de, self.atmlst)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return EmbeddingGrad(grad_method)


def kernel(embedding_obj, dm, verbose=None):
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]

    mol = embedding_obj.mol
    natoms = mol.natm
    de = np.zeros((natoms, 3))
    cppe_state = embedding_obj.cppe_state
    nuc_ee_grad = cppe_state.nuclear_interaction_energy_gradient()
    op = _grad_electrostatic_elec(mol, embedding_obj)

    # induction part
    positions = cppe_state.positions_polarizable
    n_polsites = positions.shape[0]
    if n_polsites > 0:
        nuc_field_grad = cppe_state.nuclear_field_gradient().reshape(natoms, 3, n_polsites, 3)
        mu_hf = _induced_moments(mol, embedding_obj, positions, dm, elec_only=False)
        v = _grad_induction_elec(mol, positions, mu_hf)
        grad_induction_nuc = -numpy.einsum("acpk,pk->ac", nuc_field_grad, mu_hf)
        de += grad_induction_nuc
        op += v
    elec_ee_grad = _grad_from_operator(mol, op, dm)
    de += nuc_ee_grad + elec_ee_grad
    return de
