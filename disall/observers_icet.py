from collections import namedtuple
from typing import Dict

import numpy as np

from ase import Atoms
from icet import ClusterSpace
from mchammer.observers import ClusterCountObserver
from mchammer.observers.base_observer import BaseObserver

from pymatgen.optimization.neighbors import find_points_in_spheres

ClusterCountInfo = namedtuple('ClusterCountInfo', ['counts', 'dc_tags'])

from itertools import combinations


class MCBinShortRangeOrderObserver(BaseObserver):
    def __init__(self, cluster_space, structure: Atoms,
                 radius: float, interval: int = None) -> None:
        super().__init__(interval=interval, return_type=dict,
                         tag='MCBinShortRangeOrderObserver')

        self._structure = structure
        self.r = radius
        
        self._cluster_space = ClusterSpace(
            structure=cluster_space.primitive_structure,
            cutoffs=[radius],
            chemical_symbols=cluster_space.chemical_symbols)
        
        self._cluster_count_observer = ClusterCountObserver(
            cluster_space=self._cluster_space, structure=structure,
            interval=interval)

        self._sublattices = self._cluster_space.get_sublattices(structure)
        binary_sublattice_counts = 1
#         for sublattice in self._sublattices:
#             print(sublattice)
#             print(sublattice.chemical_symbols)
            
#         for symbols in self._sublattices.allowed_species:
#             self._symbols = sorted(symbols)
        self._symbols = cluster_space.chemical_symbols[0].copy()
#         print(self._symbols)
        #### to save time
        

        structure_copy = structure.copy()
        del structure_copy[[atom.index for atom in structure_copy if atom.symbol=='B']]
        
        symbol_counts = self._get_atom_count(structure)
        
        concs = self._get_concentrations(structure)
        
        
        
        
        
        
        cart_coords = np.ascontiguousarray(np.array(structure_copy.positions), dtype=float)
        numerical_tol = 1e-8
        center_indices, neighbor_indices, images, distances = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r= self.r,
            pbc=np.array([1,1,1]),
            lattice=structure_copy.cell.array,
            tol=numerical_tol,
        )
        exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
        sender_indices, receiver_indices, images, distances = center_indices[exclude_self], \
        neighbor_indices[exclude_self], \
        images[exclude_self], \
        distances[exclude_self]
        
        bond_atom_indices = np.array([sender_indices, receiver_indices], dtype=int).T
        
        
        
        
        self.bond_atom_indices = bond_atom_indices
        
        self.n_bonds = len(self.bond_atom_indices)
        self.concs = concs
        
        
        
    def get_observable(self, structure: Atoms) -> Dict[str, float]:
        """Returns the value of the property from a cluster expansion
        model for a given atomic configurations.

        Parameters
        ----------
        structure
            input atomic structure
        """
        structure_copy = structure.copy()
        del structure_copy[[atom.index for atom in structure_copy if atom.symbol=='B']]
        
        ch_symbols = structure_copy.get_chemical_symbols()
        
        ch_atom_indices = [[ch_symbols[i[0]],ch_symbols[i[1]]] for i in self.bond_atom_indices]
        
        SRO_s = {}
        SRO_keys = []
        length = len(self._symbols)
        for i in range(length):
            for j in range(i, length):
                SRO_keys.append( (self._symbols[i], self._symbols[j]) )
        
        for ij in SRO_keys:
            if self.concs[ij[0]]>1E-12 and self.concs[ij[1]] > 1E-12:
                                                     
                prob = ch_atom_indices.count([ij[0],ij[1]]) / self.n_bonds

                SRO_s[ij] = 1 - prob/(self.concs[ij[0]]*self.concs[ij[1]])
            else:
                SRO_s[ij] = np.nan
        SRO_s['structure'] = structure.copy()
        SRO_s['structure_copy'] = structure_copy.copy()
        return SRO_s


    def _get_concentrations(self, structure: Atoms) -> Dict[str, float]:
        """Returns concentrations for each species relative its
        sublattice.

        Parameters
        ----------
        structure
            the configuration that will be analyzed
        """

        occupation = np.array(structure.get_chemical_symbols())
        concentrations = {}
        for sublattice in self._sublattices:
            if len(sublattice.chemical_symbols) == 1:
                continue
            for symbol in sublattice.chemical_symbols:
                symbol_count = occupation[sublattice.indices].tolist().count(
                    symbol)
                concentration = symbol_count / len(sublattice.indices)
                concentrations[symbol] = concentration
        return concentrations

    def _get_atom_count(self, structure: Atoms) -> Dict[str, float]:
        """Returns atom counts for each species relative its
        sublattice.

        Parameters
        ----------
        structure
            the configuration that will be analyzed
        """

        occupation = np.array(structure.get_chemical_symbols())
        counts = {}
        for sublattice in self._sublattices:
            if len(sublattice.chemical_symbols) == 1:
                continue
            for symbol in sublattice.chemical_symbols:
                symbol_count = occupation[sublattice.indices].tolist().count(
                    symbol)
                counts[symbol] = symbol_count
        return counts