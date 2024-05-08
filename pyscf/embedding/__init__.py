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

def PE(method_or_mol, solvent_obj):
    '''Initialize polarizable embedding model.

    Args:
        method_or_mol (pyscf method object or gto.Mole object)
            If method_or_mol is gto.Mole object, this function returns a
            PolEmbed object constructed with this Mole object.
        solvent_obj (PolEmbed object or dictionary with options or str)
            If solvent_obj is an object of PolEmbed class, the PE-enabled
            method will be created using solvent_obj.
            If solvent_obj is dict or str, a PolEmbed object will
            be created first with the solvent_obj, on top of which PE-enabled
            method will be created.

    Examples:

    >>> pe_options = {"potfile": "pyframe.pot"}
    >>> mf = PE(scf.RHF(mol), pe_options)
    >>> mf.kernel()
    '''
    from pyscf.embedding import embedding
    from pyscf import gto, scf, mcscf, tdscf

    if isinstance(method_or_mol, gto.mole.Mole):
        return embedding.PolarizableEmbedding(method_or_mol, solvent_obj)
    elif isinstance(method_or_mol, scf.hf.SCF):
        return embedding.embedding_for_scf(method_or_mol, solvent_obj)
