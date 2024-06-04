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
Interface to PyFraME

GitHub:      XXX
Code:        Zenodo.XXX
Publication: XXX
'''

import numpy as np

try:
    import pyframe
except ImportError:
    raise ImportError(
        'Unable to import PyFraME. Please install PyFraME.')

from pyframe.embedding import (read_input, electrostatic_interactions, induction_interactions, repulsion_interactions,
                               dispersion_interactions)

from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import df
from pyscf.embedding import _attach_embedding


@lib.with_doc(_attach_embedding._for_scf.__doc__)
def embedding_for_scf(mf, solvent_obj):
    if not isinstance(solvent_obj, PolarizableEmbedding):
        solvent_obj = PolarizableEmbedding(mf.mol, solvent_obj)
    return _attach_embedding._for_scf(mf, solvent_obj)


class EmbeddingIntegralDriver:

    def __init__(self, molecule):
        self.mol = molecule
        self.coordinates0 = None
        self.coordinates1 = None
        self.coordinates2 = None
        self.integral0 = None
        self.integral1 = None
        self.integral2_1 = None
        self.integral2_2 = None

    def electronic_fields(self,
                          coordinates: np.ndarray,
                          density_matrix: np.ndarray) -> np.ndarray:
        """Calculate the electronic fields on coordinates.

        Args:
            coordinates: Coordinates on which the fields are to be evaluated.
                Shape: (number of atoms, 3)
                Dtype: np.float64
            density_matrix: Density Matrix that is the source of the electronic field.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64

        Returns:
            Electronic fields. Shape: (number of atoms, 3) Dtype: np.float64.
        """
        if self.coordinates1 is None or not np.array_equal(self.coordinates1, coordinates):
            self.coordinates1 = coordinates
            fakemol = gto.fakemol_for_charges(self.coordinates1)
            self.integral1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
        return -1 * (np.einsum('aijg,ij->ga', self.integral1, density_matrix)
                     + np.einsum('aijg,ji->ga', self.integral1, density_matrix))

    def multipole_potential_integrals(self,
                                      multipole_coordinates: np.ndarray,
                                      multipole_orders: np.ndarray,
                                      multipoles: list[np.ndarray]) -> np.ndarray:
        """Calculate the electronic potential integrals and multiply with the multipoles.

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
            Product of electronic potential integrals and multipoles.
                Shape: (number of ao functions, number of ao functions)
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
            self.integral0 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e')
        charges = np.array([m[0:1] for m in multipoles])
        op += np.einsum('ijg,ga->ij', self.integral0 * -1.0, charges)
        # 1 order
        if np.any(multipole_orders >= 1):
            idx = np.where(multipole_orders >= 1)[0]
            dipole_coordinates = multipole_coordinates[idx]
            if self.coordinates1 is None or not np.array_equal(self.coordinates1, dipole_coordinates):
                self.coordinates1 = dipole_coordinates
                fakemol = gto.fakemol_for_charges(self.coordinates1)
                self.integral1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
            integral1 = self.integral1
            dipoles = np.array([multipoles[i][1:4] for i in idx])
            v = np.einsum('aijg,ga->ij', integral1, dipoles)
            op += v + v.T
        # 2 order
        if np.any(multipole_orders >= 2):
            idx = np.where(multipole_orders >= 2)[0]
            quadrupol_coordinates = multipole_coordinates[idx]
            if self.coordinates2 is None or not np.array_equal(self.coordinates0, quadrupol_coordinates):
                self.coordinates2 = quadrupol_coordinates
                fakemol = gto.fakemol_for_charges(quadrupol_coordinates)
                self.integral2_1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipip1')
                self.integral2_2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipvip1')
            n_sites = idx.size
            quadrupoles_non_symmetrized = np.array([multipoles[i][4:10] for i in idx])
            quadrupoles = np.zeros((n_sites, 9))
            quadrupoles[:, [0, 1, 2, 4, 5, 8]] = quadrupoles_non_symmetrized
            quadrupoles[:, [0, 3, 6, 4, 7, 8]] += quadrupoles_non_symmetrized
            quadrupoles *= -0.5
            v = np.einsum('aijg,ga->ij', self.integral2_1, quadrupoles)
            op += v + v.T
            op += np.einsum('aijg,ga->ij', self.integral2_2, quadrupoles) * 2
        return op

    def induced_dipoles_potential_integrals(self,
                                            induced_dipoles: np.ndarray,
                                            coordinates: np.ndarray) -> np.ndarray:
        """Calculate the electronic potential integrals and contract with the induced dipoles of Atoms.

        Args:
            induced_dipoles: Induced dipoles
                Shape (number of induced dipoles, 3)
                Dtype: np.float64
            coordinates: Coordinates of the induced dipoles on which the integrals are to be evaluated.
                Shape (number of induced dipoles, 3)
                Dtype: np.float64

        Returns:
            Product of the electronic potential integrals and the induced dipoles.
                Shape: (number of ao functions, number of ao functions)
                Dtype: np.float64
        """
        if self.coordinates1 is None or not np.array_equal(self.coordinates1, coordinates):
            self.coordinates1 = coordinates
            fakemol = gto.fakemol_for_charges(self.coordinates1)
            self.integral1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
        f_el_ind = np.einsum('aijg,ga->ij', -1 * self.integral1, induced_dipoles)
        return f_el_ind + f_el_ind.T

class PolarizableEmbedding(lib.StreamObject):
    _keys = {'mol', 'comm', 'options', 'classical_subsystem', 'quantum_subsystem', 'e', 'v'}

    def __init__(self, molecule, options_or_json_file):
        self.stdout = molecule.stdout
        self.verbose = molecule.verbose
        # communicator?
        self.comm = None
        # e (the electrostatic, induction energy, repulsion energy, and dispersion energy.)
        # and v (the additional potential) are
        # updated during the SCF iterations
        self.e = None
        self.v = None
        self._dm = None
        self._e_ind = None
        self._e_es = None
        self.vdw_method = None
        self.vdw_combination_rule = None

        self.mol = molecule
        self.max_memory = molecule.max_memory
        if isinstance(options_or_json_file, str):
            self.options = {"json_file": options_or_json_file}
        else:
            self.options = options_or_json_file
        if not isinstance(self.options, dict):
            raise TypeError("Options should be a dictionary.")
        self._create_pyframe_objects()
        self._integral_driver = EmbeddingIntegralDriver(molecule=self.mol)
        self._f_el_es = electrostatic_interactions.es_fock_matrix_contributions(
            classical_subsystem=self.classical_subsystem,
            integral_driver=self._integral_driver)
        self._e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(
            quantum_subsystem=self.quantum_subsystem,
            classical_subsystem=self.classical_subsystem)

        if 'vdw' in self.options:
            if not isinstance(self.options['vdw'], dict):
                raise TypeError("vdw options should be a dictionary.")
            if 'method' in self.options['vdw']:
                self.vdw_method = self.options['vdw']['method']
            else:
                self.vdw_method = 'LJ'
            if 'combination_rule' in self.options['vdw']:
                self.vdw_combination_rule = self.options['vdw']['combination_rule']
            else:
                self.vdw_combination_rule = 'Lorentz-Berthelot'
            self._e_rep = repulsion_interactions.compute_repulsion_interactions(
                quantum_subsystem=self.quantum_subsystem,
                classical_subsystem=self.classical_subsystem,
                method=self.vdw_method,
                combination_rule=self.vdw_combination_rule)
            self._e_disp = dispersion_interactions.compute_dispersion_interactions(
                quantum_subsystem=self.quantum_subsystem,
                classical_subsystem=self.classical_subsystem,
                method=self.vdw_method,
                combination_rule=self.vdw_combination_rule)
        else:
            self._e_rep = 0.0
            self._e_disp = 0.0

        if 'induced_dipoles' in self.options:
            if not isinstance(self.options['induced_dipoles'], dict):
                raise TypeError("induced_dipoles options should be a dictionary.")
            elif 'threshold' in self.options['induced_dipoles']:
                self._threshold = self.options['induced_dipoles']['threshold']
            elif 'max_iterations' in self.options['induced_dipoles']:
                self._max_iterations = self.options['induced_dipoles']['max_iterations']
            elif 'solver' in self.options['induced_dipoles']:
                self._solver = self.options['induced_dipoles']['solver']
        else:
            self._threshold = 1e-8
            self._max_iterations = 100
            self._solver = 'jacobi'

        if 'environment_energy' in self.options:
            if not isinstance(self.options['environment_energy'], bool):
                raise TypeError("environment_energy options should be a bool.")
            self._environment_energy = self.options['environment_energy']
        else:
            self._environment_energy = True

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s flags ********', self.__class__)
        for key in self.options.keys():
            logger.info(self, "pyframe.%s = %s", key, self.options[key])
        return self

    def reset(self, mol=None, options_or_json_file=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
        if options_or_json_file is not None:
            self.options = options_or_json_file
        self._create_pyframe_objects()
        self._f_el_es = None
        self._e_nuc_es = None
        self._e_rep = None
        self._e_disp = None
        self._dm = None
        self._e_ind = None
        self._e_es = None
        self.e = None
        self.v = None
        return self

    def kernel(self, dm):
        '''
        '''
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm = dm[0] + dm[1]
        self._e_ind, self._e_es, v = self._compute_pe_contributions(density_matrix=dm)
        self.e = self._e_ind + self._e_es + self._e_disp + self._e_rep
        self.v = v
        return self.e, self.v

    def nuc_grad_method(self, grad_method):
        from pyscf.embedding import embedding_gradient
        return embedding_gradient.make_grad_object(grad_method)

    def _create_pyframe_objects(self):
        # should the creation process get a callback and throw back if sth goes wrong?
        # throw exception if it is not exactly for one qm and one classical subsystem
        self.quantum_subsystem, self.classical_subsystem = (read_input.reader(
            input_data=self.options['json_file'],
            comm=self.comm))

    def _compute_pe_contributions(self, density_matrix):
        density_matrix = np.asarray(density_matrix)
        nao = density_matrix.shape[-1]
        density_matrix = density_matrix.reshape(-1, nao, nao)
        if self._e_nuc_es is None:
            self._e_nuc_es = electrostatic_interactions.compute_electrostatic_nuclear_energy(
            quantum_subsystem=self.quantum_subsystem,
            classical_subsystem=self.classical_subsystem)
        if self._f_el_es is None:
            self._f_el_es = electrostatic_interactions.es_fock_matrix_contributions(
            classical_subsystem=self.classical_subsystem,
            integral_driver=self._integral_driver)
        if self._e_rep is None or self._e_disp is None:
            if 'vdw' in self.options:
                if not isinstance(self.options['vdw'], dict):
                    raise TypeError("vdw options should be a dictionary.")
                if 'method' in self.options['vdw']:
                    self.vdw_method = self.options['vdw']['method']
                else:
                    self.vdw_method = 'LJ'
                if 'combination_rule' in self.options['vdw']:
                    self.vdw_combination_rule = self.options['vdw']['combination_rule']
                else:
                    self.vdw_combination_rule = 'Lorentz-Berthelot'
                self._e_rep = repulsion_interactions.compute_repulsion_interactions(
                    quantum_subsystem=self.quantum_subsystem,
                    classical_subsystem=self.classical_subsystem,
                    method=self.vdw_method,
                    combination_rule=self.vdw_combination_rule)
                self._e_disp = dispersion_interactions.compute_dispersion_interactions(
                    quantum_subsystem=self.quantum_subsystem,
                    classical_subsystem=self.classical_subsystem,
                    method=self.vdw_method,
                    combination_rule=self.vdw_combination_rule)
            else:
                self._e_rep = 0.0
                self._e_disp = 0.0
        e_el_es = np.einsum('ij,xij->x', self._f_el_es, density_matrix)[0]
        el_fields = self.quantum_subsystem.compute_electronic_fields(coordinates=self.classical_subsystem.coordinates,
                                                                     density_matrix=density_matrix[0],
                                                                     integral_driver=self._integral_driver)
        nuc_fields = self.quantum_subsystem.compute_nuclear_fields(self.classical_subsystem.coordinates)
        self.classical_subsystem.solve_induced_dipoles(external_fields=(el_fields + nuc_fields),
                                                       threshold=self._threshold,
                                                       max_iterations=self._max_iterations,
                                                       solver=self._solver)
        e_ind = induction_interactions.compute_induction_energy(
            induced_dipoles=self.classical_subsystem.induced_dipoles.induced_dipoles,
            total_fields=el_fields + nuc_fields + self.classical_subsystem.multipole_fields)
        f_el_ind = induction_interactions.ind_fock_matrix_contributions(classical_subsystem=self.classical_subsystem,
                                                                        integral_driver=self._integral_driver)
        return e_ind, self._e_nuc_es + e_el_es, self._f_el_es - f_el_ind
