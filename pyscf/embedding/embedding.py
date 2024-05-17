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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

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


class PolarizableEmbedding(lib.StreamObject):
    _keys = {'mol', 'comm', 'options', 'classical_subsystem', 'quantum_subsystem', 'e', 'v'}

    def __init__(self, molecule, options_or_json_file):
        self.stdout = molecule.stdout
        self.verbose = molecule.verbose
        # communicator?
        self.comm = None

        self.mol = molecule
        self.max_memory = molecule.max_memory
        if isinstance(options_or_json_file, str):
            self.options = {"json_file": options_or_json_file}
        else:
            self.options = options_or_json_file
        if not isinstance(self.options, dict):
            raise TypeError("Options should be a dictionary.")
        self._create_pyframe_objects()
        self._f_el_es = self._compute_multipole_potential()
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

        # e (the electrostatic, induction energy, repulsion energy, and dispersion energy.)
        # and v (the additional potential) are
        # updated during the SCF iterations
        self.e = None
        self.v = None
        self._dm = None
        self._e_ind = None
        self._e_es = None

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
        self.e = None
        self.v = None
        self._dm = None
        self._e_ind = None
        self._e_es = None
        return self

    def kernel(self, dm):
        '''
        '''
        if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm = dm[0] + dm[1]
        self._e_ind, self._e_es, v = self._compute_pe_contributions(density_matrices=dm)
        self.e = self._e_ind + self._e_es + self._e_disp + self._e_rep
        self.v = v
        return self.e, self.v

    def _create_pyframe_objects(self):
        # should the creation process get a callback and throw back if sth goes wrong?
        # throw exception if its not exactly for one qm and one classical subsystem
        self.quantum_subsystem, self.classical_subsystem = (read_input.reader(
            input_data=self.options['json_file'],
            comm=self.comm))

    def _compute_multipole_potential(self):
        if np.any(self.classical_subsystem.multipole_orders > 2):
            raise NotImplementedError("""Multipole potential integrals not
                                      implemented for order > 2.""")
        moments = self.classical_subsystem.degenerate_multipoles_with_taylor_coefficients
        op = 0
        # op -= because sign is different compared to vlx?
        # 0 order
        fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates)
        integral0 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e')
        moments_0 = np.array([m[0:1] for m in moments])
        op -= np.einsum('ijg,ga->ij', integral0, moments_0)
        # 1 order
        if np.any(self.classical_subsystem.multipole_orders >= 1):
            idx = np.where(self.classical_subsystem.multipole_orders >= 1)[0]
            fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates[idx])
            integral1 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
            moments_1 = np.array([moments[i][1:4] for i in idx])
            v = np.einsum('aijg,ga->ij', integral1, moments_1)
            op -= v + v.T
        # 2 order
        if np.any(self.classical_subsystem.multipole_orders >= 2):
            idx = np.where(self.classical_subsystem.multipole_orders >= 2)[0]
            fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates[idx])
            n_sites = idx.size
            moments_2_non_symmetrized = np.array([moments[i][4:10] for i in idx])
            moments_2 = np.zeros((n_sites, 9))
            # FIXME build my own matrix.
            moments_2[:, [0, 1, 2, 4, 5, 8]] = moments_2_non_symmetrized
            moments_2[:, [0, 3, 6, 4, 7, 8]] += moments_2_non_symmetrized
            moments_2 *= .5
            integral2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipip1')
            v = np.einsum('aijg,ga->ij', integral2, moments_2)
            op -= v + v.T
            integral2 = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ipvip1')
            op -= np.einsum('aijg,ga->ij', integral2, moments_2) * 2
        return op

    def _compute_pe_contributions(self, density_matrices):
        density_matrices = np.asarray(density_matrices)
        nao = density_matrices.shape[-1]
        density_matrices = density_matrices.reshape(-1, nao, nao)
        e_el_es = np.einsum('ij,xij->x', self._f_el_es, density_matrices)[0]
        fakemol = gto.fakemol_for_charges(self.classical_subsystem.coordinates)
        # first order derivative of the electronic potential integral
        j3c = df.incore.aux_e2(self.mol, fakemol, intor='int3c2e_ip1')
        electric_fields = (np.einsum('aijg,ij->ga', j3c, density_matrices[0]) +
                           np.einsum('aijg,ji->ga', j3c, density_matrices[0]))
        nuclear_fields = self.quantum_subsystem.compute_nuclear_fields(self.classical_subsystem.coordinates)
        # Set ind dipoles options here
        self.classical_subsystem.solve_induced_dipoles(external_fields=(-electric_fields + nuclear_fields),
                                                       threshold=self._threshold,
                                                       max_iterations=self._max_iterations,
                                                       solver=self._solver)
        e_ind = induction_interactions.compute_induction_energy(
            induced_dipoles=self.classical_subsystem.induced_dipoles.
            induced_dipoles,
            total_fields=-electric_fields + nuclear_fields +
                         self.classical_subsystem.multipole_fields)
        f_el_ind = np.einsum('aijg,ga->ij', j3c, self.classical_subsystem.induced_dipoles.
                             induced_dipoles)
        f_el_ind = f_el_ind + f_el_ind.T
        return e_ind, self._e_nuc_es + e_el_es, self._f_el_es - f_el_ind
