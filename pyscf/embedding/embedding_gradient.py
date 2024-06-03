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

try:
    import pyframe
except ImportError:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.')

from pyframe.embedding import (electrostatic_interactions, induction_interactions, repulsion_interactions,
                               dispersion_interactions)

from pyscf.lib import logger
from pyscf import gto
from pyscf import df, scf
from pyscf.embedding._attach_embedding import _Embedding
from pyscf.grad import rhf as rhf_grad


class EmbeddingIntegralDriver:

    def __init__(self, molecule):
        self.mol = molecule
        self.coordinates0 = None
        self.coordinates1 = None
        self.integral0 = None
        self.integral1_1 = None
        self.integral1_2 = None

    def multipole_potential_gradient_integrals(self,
                                               multipole_coordinates: np.ndarray,
                                               multipole_orders: np.ndarray,
                                               multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the gradient of the electronic potential integrals and multiply with the multipoles.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64.
            multipole_orders: Multipole orders of all multipoles.
                Shape: (number of atoms)
                Dtype: np.int64
            multipoles: Multipoles multiplied with degeneracy coefficients and taylor coefficients.
                Shape: (number of atoms, number of multipole elements)
                Dtype: np.float64

        Returns:
            Product of gradient of electronic potential integrals and multipoles.
                Shape: (number of nuclei, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        if np.any(multipole_orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                             implemented for order > 2.""")
        op = 0
        # 0 order
        idx = np.where(multipole_orders >= 0)[0]
        charge_coordinates = multipole_coordinates[idx]
        if self.coordinates0 is None or not np.array_equal(self.coordinates0, charge_coordinates):
            self.coordinates0 = charge_coordinates
            fakemol = gto.fakemol_for_charges(charge_coordinates)
            self.integral0 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1', comp=3)
        charges = np.array([m[0:1] for m in multipoles])
        op += np.einsum('cijg,ga->cij', self.integral0 * -1.0, charges)
        # 1 order
        if np.any(multipole_orders >= 1):
            idx = np.where(multipole_orders >= 1)[0]
            dipole_coordinates = multipole_coordinates[idx]
            if self.coordinates1 is None or not np.array_equal(self.coordinates1, dipole_coordinates):
                self.coordinates1 = dipole_coordinates
                fakemol = gto.fakemol_for_charges(self.coordinates1)
                self.integral1_1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipip1', comp=9)
                self.integral1_2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipvip1', comp=9)
            integral1_1 = self.integral1_1
            integral1_2 = self.integral1_2
            integral1_1 = integral1_1.reshape(3, 3, *integral1_1.shape[1:])
            integral1_2 = integral1_2.reshape(3, 3, *integral1_2.shape[1:])
            dipoles = np.array([multipoles[i][1:4] for i in idx])
            v = np.einsum('caijg,ga->cij', integral1_1, dipoles)
            v += np.einsum('caijg,ga->cij', integral1_2, dipoles)
            op += v
        # 2 order
        if np.any(multipole_orders >= 2):
            idx = np.where(multipole_orders >= 2)[0]
            n_sites = idx.size
            quadrupoles_non_symmetrized = np.array([multipoles[i][4:10] for i in idx])
            quadrupoles = np.zeros((n_sites, 9))
            # FIXME build my own matrix.
            quadrupoles[:, [0, 1, 2, 4, 5, 8]] = quadrupoles_non_symmetrized
            quadrupoles[:, [0, 3, 6, 4, 7, 8]] += quadrupoles_non_symmetrized
            quadrupoles *= -0.5
            quadrupol_coordinates = multipole_coordinates[idx]
            for ii, pos in enumerate(quadrupol_coordinates):
                with self.mol.with_rinv_orig(pos):
                    int1 = self.mol.intor('int1e_ipipiprinv', comp=27).reshape(3, 9, self.mol.nao, self.mol.nao)
                    int2 = self.mol.intor('int1e_ipiprinvip', comp=27).reshape(3, 9, self.mol.nao, self.mol.nao)
                    int3 = int2.reshape(3, 3, 3, -1).transpose(2, 0, 1, -1).reshape(3, 9, self.mol.nao, self.mol.nao)
                    op += np.einsum('caij,a->cij', int1, quadrupoles[ii])
                    op += np.einsum('caji,a->cij', int3, quadrupoles[ii])
                    op += 2.0 * np.einsum('caij,a->cij', int2, quadrupoles[ii])
        return op

    def induced_fock_matrix_contributions_gradient(self,
                                                   multipole_coordinates: np.ndarray,
                                                   induced_dipoles: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the induced Fock-matrix contributions.

        Args:
            multipole_coordinates: Coordinates of the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            induced_dipoles: Induced dipoles on the Multipoles.
                Shape: (number of atoms, 3)
                Dtype: np.float64

        Returns:
            Gradient of induced Fock-matrix contributions.
                Shape: (number of nuclei, number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        if self.coordinates1 is None or not np.array_equal(self.coordinates1, multipole_coordinates):
            self.coordinates1 = multipole_coordinates
            fakemol = gto.fakemol_for_charges(self.coordinates1)
            self.integral1_1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipip1', comp=9)
            self.integral1_2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipvip1', comp=9)
        integral1_1 = self.integral1_1
        integral1_2 = self.integral1_2
        integral1_1 = integral1_1.reshape(3, 3, *integral1_1.shape[1:])
        integral1_2 = integral1_2.reshape(3, 3, *integral1_2.shape[1:])
        v = np.einsum('caijg,ga->cij', integral1_1, induced_dipoles)
        v += np.einsum('caijg,ga->cij', integral1_2, induced_dipoles)
        return v


def make_grad_object(grad_method):
    '''
    '''
    assert isinstance(grad_method.base, _Embedding)
    logger.warn(grad_method, "PE gradients are not optimized for performance.")
    if not isinstance(grad_method.base, scf.hf.SCF):
        raise NotImplementedError("PE gradients only implemented for SCF methods.")
    grad_method_class = grad_method.__class__

    class EmbeddingGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_classical_subsystem = None
            self.de_quantum_subsystem = None
            self._keys = self._keys.union(['de_classical_subsystem', 'de_quantum_subsystem'])


        # TODO: if moving to python3, change signature to
        # def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        def kernel(self, *args, **kwargs):
            '''
            '''
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = self.base.make_rdm1(ao_repr=True)

            self.de_classical_subsystem = kernel(self.base.with_embedding, dm)
            self.de_quantum_subsystem = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_quantum_subsystem + self.de_classical_subsystem

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
    quantum_subsystem = embedding_obj.quantum_subsystem
    classical_subsystem = embedding_obj.classical_subsystem
    e_es_nuc_grad = electrostatic_interactions.compute_electrostatic_nuclear_gradients(
        quantum_subsystem=quantum_subsystem,
        classical_subsystem=classical_subsystem)
    integral_driver = EmbeddingIntegralDriver(molecule=mol)
    f_es_grad = electrostatic_interactions.es_fock_matrix_gradient_contributions(
        classical_subsystem=classical_subsystem,
        integral_driver=integral_driver)
    e_es_el_grad = _grad_from_operator(mol, f_es_grad, dm)
    el_fields = quantum_subsystem.compute_electronic_fields(coordinates=classical_subsystem.coordinates,
                                                                 density_matrix=dm,
                                                                 integral_driver=embedding_obj._integral_driver)
    nuc_fields = quantum_subsystem.compute_nuclear_fields(classical_subsystem.coordinates)
    classical_subsystem.solve_induced_dipoles(external_fields=(el_fields + nuc_fields),
                                              threshold=embedding_obj._threshold,
                                              max_iterations=embedding_obj._max_iterations,
                                              solver=embedding_obj._solver)
    nuc_field_grad = quantum_subsystem.compute_nuclear_field_gradients(coordinates=classical_subsystem.coordinates)
    f_ind_grad = induction_interactions.induced_fock_matrix_contributions_gradient(
        classical_subsystem=classical_subsystem,
        integral_driver=integral_driver)
    e_ind_el_grad = _grad_from_operator(mol, f_ind_grad, dm)
    e_ind_nuc_grad = induction_interactions.compute_induction_energy_gradient(
        induced_dipoles=classical_subsystem.induced_dipoles.induced_dipoles,
        total_field_gradients=nuc_field_grad)
    if embedding_obj.vdw_method is not None:
        e_rep_grad = repulsion_interactions.compute_repulsion_interactions_gradient(
            quantum_subsystem=quantum_subsystem,
            classical_subsystem=classical_subsystem,
            method=embedding_obj.vdw_method,
            combination_rule=embedding_obj.vdw_combination_rule)
        e_disp_grad = dispersion_interactions.compute_dispersion_interactions_gradient(
            quantum_subsystem=quantum_subsystem,
            classical_subsystem=classical_subsystem,
            method=embedding_obj.vdw_method,
            combination_rule=embedding_obj.vdw_combination_rule)
    else:
        e_rep_grad = 0.0
        e_disp_grad = 0.0
    de += -e_ind_nuc_grad  + e_ind_el_grad - e_es_nuc_grad + e_es_el_grad + e_rep_grad + e_disp_grad
    return de

def _grad_from_operator(mol, op, dm):
    natoms = mol.natm
    ao_slices = mol.aoslice_by_atom()
    grad = np.zeros((natoms, 3))
    for ia in range(natoms):
        k0, k1 = ao_slices[ia, 2:]
        Dx_a = np.zeros_like(op)
        Dx_a[:, k0:k1] = op[:, k0:k1]
        Dx_a += Dx_a.transpose(0, 2, 1)
        grad[ia] -= np.einsum("xpq,pq", Dx_a, dm)
    return grad
