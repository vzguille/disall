import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms








def get_info_from_DFT(row, calculator_label, try_no):
    max_sc_steps = 60
    num_sc_steps = np.array([len(i) for i in row[calculator_label]['energy_electronic_step_{:02d}'.format(try_no)] ])
    structures_traj = []
    energy_traj = []
    energy_per_atom_traj = []
    force_traj = []
    stress_traj = []
    converged_index = np.where(num_sc_steps < max_sc_steps)[0].tolist()
    negative_index = np.where (row[calculator_label]['energy_traj_{:02d}'.format(try_no)] < 0)[0].tolist()
    intersection = list(set(converged_index) & set(negative_index))

    size = len(row['init_structure'])
    
    for i in intersection:
        structures_traj.append(row[calculator_label]['structures_traj_{:02d}'.format(try_no)][i])
        energy_traj.append(row[calculator_label]['energy_traj_{:02d}'.format(try_no)][i])
        energy_per_atom_traj.append(row[calculator_label]['energy_traj_{:02d}'.format(try_no)][i]/size)
        force_traj.append(row[calculator_label]['force_traj_{:02d}'.format(try_no)][i])
        stress_traj.append(row[calculator_label]['stress_traj_{:02d}'.format(try_no)][i])
    
    return structures_traj,  energy_traj, energy_per_atom_traj, force_traj, stress_traj
        
def get_indexes_per_calculation(df_func, calculator_label):
    data = df_func.apply(get_info_from_DFT, args = (calculator_label, 0,), axis = 1)
    list1, list2, list3, list4, list5 = [], [], [], [], []
    index_list = []  # Stores original indices
    
    # Iterate through the series
    for idx, (l1, l2, l3, l4, l5) in data.items():
        list1.extend(l1)
        list2.extend(l2)
        list3.extend(l3)
        list4.extend(l4)
        list5.extend(l5)
        
        # Append the original index for each item in the lists
        index_list.extend([idx] * len(l1))  # Same length as any of the lists
    
    # Convert to pandas DataFrame if needed
    df = pd.DataFrame({
        'Index': index_list,
        'structure': list1,
        'energy': list2,
        'energy_per_atom': list3,
        'force': list4,
        'stress': list5
    })
    
    return df

def _last_step_is_negative(row, calculator_label, try_no):
    return row[calculator_label]['energy_traj_{:02d}'.format(try_no)][-1] < 0
def _last_step_is_converged(row, calculator_label, try_no):
    return len(row[calculator_label]['energy_electronic_step_{:02d}'.format(try_no)][-1]) < 60


def get_split(df_func, calculator_label, size = 0, train_split = 0.7, try_no = 0, random_state = None):
    series_negative = df_func.apply(_last_step_is_negative, args = (calculator_label, try_no,), axis = 1)
    series_converge = df_func.apply(_last_step_is_converged, args = (calculator_label, try_no,), axis = 1)
    filtered_bool = (series_converge & series_negative)
    filtered_index = filtered_bool.index[filtered_bool].tolist()

    outer_diff = list(set(df_func.index.tolist()) ^ set(filtered_index))
    # print(outer_diff)
    # print(filtered_index)
    if train_split < 1E-8:
        return filtered_index
    if size == 0:
        split_01_index, split_02_index = train_test_split(filtered_index, train_size= train_split, random_state = random_state)
        split_01_index.sort()
        split_02_index.sort()

    if size > 0:
        split_01_index = list(set(df_func[df_func['size'] < size].index.tolist()) & set(filtered_index))
        to_split = list(set(split_01_index) ^ set(filtered_index))
        plus_01_index, split_02_index = train_test_split(to_split, train_size= train_split, random_state = random_state)
        split_01_index = split_01_index + plus_01_index
        split_01_index.sort()
        split_02_index.sort()
    return filtered_index, split_01_index, split_02_index

def df_to_chgnet(df_func):
    structures = df_func['structure'].apply(ase_to_pmg).tolist()
    energies = df_func['energy_per_atom'].tolist()
    forces = df_func['force'].tolist()
    stresses = df_func['stress'].tolist()
    return structures, energies, forces, stresses


def from_calc_to_static_indexes(df_, train_index, test_index, validation_index):
    
    train_static_index = df_[df_['Index'].isin(train_index)].index.tolist()
    test_static_index = df_[df_['Index'].isin(test_index)].index.tolist()
    validation_static_index = df_[df_['Index'].isin(validation_index)].index.tolist()
    
    all_index = train_static_index + test_static_index + validation_static_index
    all_index.sort()
    
    train_static_index = [all_index.index(entry) for entry in train_static_index if entry in all_index]
    test_static_index = [all_index.index(entry) for entry in test_static_index if entry in all_index]
    validation_static_index = [all_index.index(entry) for entry in validation_static_index if entry in all_index]
    
    return train_static_index, test_static_index, validation_static_index
    
######


######

def get_lims(xs, ys, panf=0.05):
    
    h_=np.append(xs, ys)
    mi,ma=np.min(h_),np.max(h_)
    pan=panf*(ma-mi)
    return mi-pan,ma+pan

def parity_plot(ax, gtcl_list,s=5,colors=False,color='blue',unit='',alpha=1, xlabel='', ylabel=''):
    '''parity plot but only for one test'''
    
    
    ii=gtcl_list[0][0]
    jj=gtcl_list[0][1]

    if colors is not False:
        ax.scatter(ii, jj, s=s, c=colors, alpha=alpha)
    else:    
        ax.scatter(ii, jj, s=s, color=color, alpha=alpha)
    mi,ma=get_lims(ii, jj)
    ax.set_xlim(mi,ma)
    ax.set_ylim(mi,ma)
    ax.grid(True)
    
    ax.annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii,jj)),xy=(0.1,0.9),xycoords='axes fraction')
    ax.annotate(r'$RMSE=$'+'{:.4f}{}'.format(np.sqrt(mean_squared_error(ii,jj)),unit),xy=(0.1,0.8),xycoords='axes fraction')
    ax.annotate(r'$MAE=$'+'{:.4f}{}'.format(mean_absolute_error(ii,jj), unit),xy=(0.1,0.7),xycoords='axes fraction')
    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)

def validation_static(model, ground_truth, plot = True):
    structures, energies, forces, stresses = ground_truth
    pred_energies = []
    pred_forces = []
    pred_stresses = []
    for i, ii in enumerate(structures):
        res = model.predict_structure(structures[i].copy())
        pred_energies.append(float(res['e']))
        pred_forces.append(res['f'])
        pred_stresses.append(res['s'])

    pred_forces = np.concatenate([i.ravel() for i in pred_forces])
    pred_stresses = np.array(pred_stresses).ravel()
    
    forces_gt = np.concatenate([i.ravel() for i in forces])
    stresses_gt = np.array([i*-0.1 for i in stresses]).ravel()
    
    
    if plot:
        # Create the figure and 1x3 subplot grid
        fig, axes = plt.subplots(1, 3, figsize=(5*3, 5), sharex=False, sharey=False)
        
        
        parity_plot(axes[0], [[energies, pred_energies]], unit =r'$eV/atom$', xlabel='DFT Energy', ylabel='CHGNet Energy')
        parity_plot(axes[1], [[forces_gt, pred_forces]], unit =r'$eV/A \dot atom$', xlabel='DFT Force', ylabel='CHGNet Force')
        parity_plot(axes[2], [[stresses_gt, pred_stresses]], unit =r'$GPa$', xlabel='DFT Stress', ylabel='CHGNet Stress')
        
        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
        return pred_energies, pred_forces, pred_stresses
    else:
        return [[energies, pred_energies]], [[forces_gt, pred_forces]], [[stresses_gt, pred_stresses]]





def ase_checkrelax(ase_ini,ase_fin,cutoff=0.05):
    if not isinstance(ase_ini, np.ndarray):
        before = ase_ini.get_cell().array
    else:
        before = ase_ini.copy()
    if not isinstance(ase_fin, np.ndarray):
        after  = ase_fin.get_cell().array
    else:
        after = ase_fin.copy()
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

def df_to_relax(df, list_index, calculator_label, try_no = 0):
    init_structure = []
    energy = []
    volume_final = []
    strain_final = []
    for i in list_index:
        init_structure.append(df.loc[i, 'init_structure'])
        energy.append(df.loc[i, calculator_label]['energy_traj_{:02d}'.format(try_no)][-1]/len(init_structure[-1]))
        volume_final.append(df.loc[i, calculator_label]['structures_traj_{:02d}'.format(try_no)][-1].get_volume()/len(init_structure[-1]))
        strain_final.append(ase_checkrelax(init_structure[-1], df.loc[i, 'first_00']['structures_traj_{:02d}'.format(try_no)][-1] ))    
    return init_structure, energy, volume_final, strain_final

def validation_relax(optimizer, ground_truth, fmax = 0.01, plot = True):
    structures, energies, volumes, strains = ground_truth
    pred_energies = []
    pred_volumes = []
    pred_strains = []
    
    for i, ii in enumerate(structures):
        res_it = optimizer.relax(structures[i].copy(), fmax = fmax)
        
        pred_energies.append( res_it["trajectory"].energies[-1]/len(structures[i]) )
        
        cell = res_it["trajectory"].cells[-1]
        pred_volumes.append(
            np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))/len(structures[i]))
        
        
        pred_strains.append(ase_checkrelax(structures[i].copy(), res_it["trajectory"].cells[-1]))

    if plot:
        # Create the figure and 1x3 subplot grid
        fig, axes = plt.subplots(1, 3, figsize=(5*3, 5), sharex=False, sharey=False)
        
        
        parity_plot(axes[0], [[energies, pred_energies]], unit =r'$eV/atom$', xlabel='DFT relaxed Energy', ylabel='CHGNet-LBFGS relaxed Energy')
        parity_plot(axes[1], [[volumes, pred_volumes]], unit =r'$A^3$', xlabel='DFT relaxed Volume', ylabel='CHGNet-LBFGS relaxed Volume')
        parity_plot(axes[2], [[strains, pred_strains]], unit =r'', xlabel=r'DFT relaxed $\epsilon$', ylabel=r'CHGNet-LBFGS relaxed $\epsilon$')
        
        # Adjust layout and show the figure
        plt.tight_layout()
        plt.show()
        return pred_energies, pred_volumes, pred_strains
    else:
        return [[energies, pred_energies]], [[volumes, pred_volumes]], [[strains, pred_strains]]
    
