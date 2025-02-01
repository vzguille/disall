from collections import Counter
import numpy as np
import pandas as pd
import itertools

from ase.build.supercells import make_supercell
from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def ase_checkrelax(ase_ini,ase_fin,cutoff=0.05):
    before = ase_ini.get_cell().array
    after  = ase_fin.get_cell().array
    before_res = before/np.power(np.linalg.det(before), 1./3.)
    after_res  = after/np.power(np.linalg.det(after), 1./3.)
    diff = np.dot(np.linalg.inv(before_res), after_res)

    out_mat = (diff + diff.T)/2 - np.eye(3)
    distorsion_val = np.linalg.norm(out_mat)
    # os.system(f"echo {distorsion_val} > distorsion")
    #if distorsion_val > cutoff:
        #os.system(f"touch error")
        #print('error')
    return distorsion_val

def row_comp(x):
    old_dict = Counter(x['init_structure'].get_chemical_symbols())
    return {key: value / x['size'] for key, value in old_dict.items()}
    
def row_reference(x, dictionary_reference, calculator_label):
    value_reference = x[calculator_label][1]/len(x['init_structure'])
    for i in dictionary_reference.keys():
        value_reference = value_reference- (x[i]*dictionary_reference[i])
    return value_reference

def row_strain(x , calculator_label):
    return ase_checkrelax(x['init_structure'], x[calculator_label][0])

def row_volume(x):
    return x['structure_final'].get_volume()/x['size']


def parse_to_ce(df, calculator_label):

    num_atoms = len(df[df['size'] == 1])

    dict_reference = {}
    for i, ii in df.iterrows():
        if i == num_atoms: break
        dict_reference[ii['init_structure'].get_chemical_symbols()[0]] = df.loc[i,calculator_label][1]

    data = df.apply(row_comp, axis=1)
    added = pd.DataFrame.from_records(data).fillna(0)
    relax_chgnet_df = pd.concat([df.reset_index(drop=False), added.reset_index(drop=True)], axis=1)
    relax_chgnet_df = relax_chgnet_df.set_index('index', drop = True)

    relax_chgnet_df['energy_reference']= relax_chgnet_df.apply(row_reference, 
                    args=(dict_reference, calculator_label), 
                    axis=1)
    relax_chgnet_df['strain_level']= relax_chgnet_df.apply(row_strain,
                    args=(calculator_label,), 
                    axis=1)
    relax_chgnet_df = relax_chgnet_df.sort_index()

    elements = list(dict_reference.keys())
    elements.sort()

    prim = df.loc[0, 'init_structure'].copy()
    
    sgn = SpacegroupAnalyzer(ase_to_pmg(prim)).get_space_group_number()
    if sgn == 225:
        conventional_parameter = np.sqrt(2) * np.linalg.norm(prim.cell.array[0])

    return relax_chgnet_df, prim, conventional_parameter, elements

def filter_strain(df, strain_treshold = 0.05):
    # df_work_set
    df_work = df.copy()
    df_work = df_work[df_work['strain_level'] < strain_treshold]
    df_work = df_work.sort_values(by = 'index')
    return df_work

###

def make_supercell_cubic(prim, times = [5, 5, 8], conventional = False):
    if conventional:
        sgn = SpacegroupAnalyzer(ase_to_pmg(prim)).get_space_group_number()
        if sgn == 225:
            conv_a0 =  np.sqrt(2) * np.linalg.norm(prim.cell.array[0])
        conv_cell = np.array([[conv_a0,0,0],[0,conv_a0,0],[0,0,conv_a0]])
        MM = conv_cell@ np.linalg.inv(prim.cell.array)

        conv_structure = make_supercell(prim, MM)
        structure = conv_structure.copy()
    else:
        structure = prim.copy()
    ce_supercell = make_supercell(structure, np.array([[times[0],0,0],
                                             [0,times[1],0],
                                             [0,0,times[2]],]))
    return ce_supercell

def best_integer_composition(compositions, min_l = 6, max_l = 10, conventional = False, prim = None):
    # Step 1: Compute all unique (m, n, k) values and their corresponding sums
    
    
    if conventional:
        sgn = SpacegroupAnalyzer(ase_to_pmg(prim)).get_space_group_number()
        if sgn == 225:
            conv_a0 =  np.sqrt(2) * np.linalg.norm(prim.cell.array[0])
        conv_cell = np.array([[conv_a0,0,0],[0,conv_a0,0],[0,0,conv_a0]])
        MM = conv_cell@ np.linalg.inv(prim.cell.array)
        conv_structure = make_supercell(prim, MM)
        size_conv = len(conv_structure)
    else:
        size_conv = 1
    possible_sums = []
    for m, n, k in itertools.combinations_with_replacement(range(min_l, max_l), 3):
        possible_sums.append((size_conv * m * n * k, m, n, k))


    # Step 2: Normalize the composition to sum to 1
    compositions = np.array(compositions)
    normalized = compositions / np.sum(compositions)

    best_sum = None
    best_int_values = None
    best_mnk = None
    min_error = float('inf')

    # Step 3: Iterate over all possible sums and find the best one
    for target_sum, m, n, k in possible_sums:
        # Scale composition to this sum
        scaled_values = normalized * target_sum
        floored_values = np.floor(scaled_values).astype(int)  # Initial floor rounding
        
        # Compute remainders (how much each value was reduced by flooring)
        remainders = scaled_values - floored_values
        
        # Compute the deficit (how much we need to adjust to meet the exact target sum)
        deficit = target_sum - np.sum(floored_values)
        
        # Fix sum by distributing the deficit among the largest remainders
        indices = np.argsort(remainders)[::-1]  # Sort indices by largest remainder
        for i in range(deficit):
            floored_values[indices[i]] += 1  # Increase values with highest remainders first
        
        # Compute total error (sum of absolute differences)
        error = np.sum(np.abs( (floored_values/np.sum(floored_values)) - normalized))
        
        # Select the best sum that minimizes the error
        if error < min_error - 1E-8: # ensure we get the smallest structure
            min_error = error
            best_sum = target_sum
            best_mnk = [m, n, k]
            best_int_values = floored_values

    return best_int_values.tolist(), best_sum, best_mnk, min_error