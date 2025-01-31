from collections import Counter
import numpy as np
import pandas as pd



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


def parse_relaxer(df, calculator_label):

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
    return relax_chgnet_df, df.loc[0, 'init_structure'].copy(), elements

def filter_strain(df, strain_treshold = 0.05):
    # df_work_set
    df_work = df.copy()
    df_work = df_work[df_work['strain_level'] < strain_treshold]
    df_work = df_work.sort_values(by = 'index')
    return df_work