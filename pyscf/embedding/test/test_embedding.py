#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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

import unittest
import os
import pyscf

try:
    import pyframe
    from pyscf import embedding
    from pyscf import scf
except ImportError:
    pass

dir_name = os.path.dirname(__file__)

class TestPolEmbed(unittest.TestCase):
    def test_pol_embed_scf(self):
        mol = pyscf.M(atom='''
        C     32.081000    29.944000    30.110000
        C     30.771000    30.374000    29.820000
        C     29.571000    29.984000    30.590000
        C     28.361000    30.394000    30.239000
        H     32.911000    30.224000    29.460000
        H     32.181000    29.274000    30.950000
        H     30.701000    31.104000    29.020000
        H     29.571000    29.434000    31.540000
        H     28.241000    31.044000    29.380000
        H     27.441000    30.054000    30.720000
        ''', basis='sto3g', verbose=4)
        mf = embedding.PE(scf.RHF(mol), 'butadiene_water.json')
        mf.conv_tol = 1e-10
        mf.kernel()
        ref_pe_energy = -0.006102943337
        ref_scf_energy = -153.004117624242
        self.assertAlmostEqual(mf.with_embedding.e, ref_pe_energy, 8)
        self.assertAlmostEqual(mf.e_tot, ref_scf_energy, 8)

if __name__ == "__main__":
    print("Full Tests for pol_embed")
    unittest.main()
