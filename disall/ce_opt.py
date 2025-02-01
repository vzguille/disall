import os
import shutil
import time
import copy

import numpy as np
import datetime
import pickle

from disall.observers_icet import MCBinShortRangeOrderObserver

import matplotlib.pyplot as plt

from mchammer.ensembles import CanonicalEnsemble
from mchammer.calculators import ClusterExpansionCalculator
from mchammer import DataContainer

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C

from skopt.acquisition import gaussian_ei

scaler = StandardScaler



def get_args_opt(ELE_LIST, integer_compositions_ite, directory = '.'):

    concentration = {}
    string_concentration = ''
    for j, jj in enumerate(ELE_LIST[0]):
        concentration[jj] = integer_compositions_ite[j]
        string_concentration += jj+'{:04d}'.format(integer_compositions_ite[j])
    SRO_keys = []
    length = len(ELE_LIST[0])
    for i in range(length):
        for j in range(i, length):
            SRO_keys.append( (ELE_LIST[0][i], ELE_LIST[0][j]))
    return {'concentration': concentration,
            'string_concentration':string_concentration,
            'SRO_keys': SRO_keys,
            'directory_logs':directory+'/'+string_concentration+'/',
           }



def recreate_directory(directory_path, delete = False):
    
    if os.path.exists(directory_path):
        if delete:
            shutil.rmtree(directory_path)
    os.makedirs(directory_path)            

def log_message(message, log_file='app.log'):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'{timestamp} - {message}\n')
    with open(log_file, 'a') as f:
        f.write(f'{timestamp} - {message}\n')


def get_a0(prim):
    sgn = SpacegroupAnalyzer(ase_to_pmg(prim)).get_space_group_number()
    if sgn == 225:
        conventional_parameter = np.sqrt(2) * np.linalg.norm(prim.cell.array[0])
    return conventional_parameter

def plot_GP(gp, X, y, scaler, highlight = 0, save = False, metadata = {}):
    X_pred = np.linspace(-2.2, 3, 1500).reshape(-1, 1)
    y_pred, sigma = gp.predict(X_pred, return_std=True)
    
    plt.figure()

    
    colors = ['y']*len(X)
    if highlight > 0:
        colors[-highlight:] = 'r'

    realX = scaler.inverse_transform(X)
    realX_pred = scaler.inverse_transform(X_pred)

    plt.scatter(realX, y, color = colors, s=50, label='Ground Truth')
    plt.plot(realX_pred, y_pred, 'k-', label='GP prediction')
    plt.fill_between(realX_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression')
    plt.legend()
    if save:
        title = 'subsystem_{}_iteration_{:02d}'.format(
            ''.join(metadata['subset']),
            metadata['iteration'],)
        # title = 'subsystem_{}_iteration_{:02d}_temperature_{:09.2f}'.format(
        #     metadata['subset'],
        #     metadata['iteration'],
        #     metadata['temp'],)
        plt.title(title)
        plt.savefig(metadata['directory'] + title, dpi=150, bbox_inches='tight')
    plt.show()

def plot_entropy(temps, entropy, ideal_entropy = 0, save = False, metadata = {}):
    fig, ax = plt.subplots()

    
    
    
    ax.scatter(temps, entropy, s=50, label='Entropy')
    if ideal_entropy > 0:
        ax.plot([-500, 4000], [0.95*ideal_entropy, 0.95*ideal_entropy], color='r', linestyle='--')    
        ax.set_ylim(0, ideal_entropy*1.3)
    ax.set_xlim(0, 3400)
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Entropy')
    ax.legend()
    if save:
        title = 'subsystem_{}_entropy_iteration_{:02d}'.format(
            ''.join(metadata['subset']),
            metadata['iteration'])
        plt.title(title)
        plt.savefig(metadata['directory'] + title, dpi=150, bbox_inches='tight')
    plt.show()




def plot_GP_and_entropy(gp, X, y, scaler, entropy, ideal_entropy = 0, alpha = 0.5, highlight = 0, save = False, metadata ={}):
    #
    X_pred = np.linspace(-2.2, 3, 1500).reshape(-1, 1)
    y_pred, sigma = gp.predict(X_pred, return_std=True)
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 8), sharex=True)
    colors = ['y']*len(X)
    if highlight > 0:
        colors[-highlight:] = 'r'

    realX = scaler.inverse_transform(X)
    realX_pred = scaler.inverse_transform(X_pred)
    
    ax1.scatter(realX, y, color = colors, s=50, label='Ground Truth')
    ax1.grid()
    ax1.plot(realX_pred, y_pred, 'k-', label='GP prediction')
    ax1.fill_between(realX_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color='k')
    ax1.set_xlim(0, 4000)
    ax1.set_ylim(-0.5, 2.0)
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Obj. function')
    ax1.legend()

    
    ax2.scatter(realX, entropy, color = colors, edgecolor = 'red', s=50, alpha = alpha, label='Entropy')
    if ideal_entropy > 0:
        ax2.plot([-500, 4000], [0.95*ideal_entropy, 0.95*ideal_entropy], color='r', linestyle='--')    
        ax2.set_xlim(0, 4000)
        ax2.set_ylim(0, ideal_entropy*1.3)

    ax2.grid()
    ax2.set_xlabel('Temps')
    ax2.set_ylabel(r'Entropy $k_b$/atom')
    fig.subplots_adjust(hspace=0)

    if save:
        title = '{}, iteration {:03d}'.format(
            ''.join(metadata['subset']),
            metadata['iteration'],)
        fig.suptitle(title)
        fig.savefig(metadata['directory'] + 'gp_and_entropy_{}_{:03d}'.format(
                            ''.join(metadata['subset']),
                            metadata['iteration'],), 
                    bbox_inches='tight', 
                    pad_inches=0.1, 
                    dpi=150)

    plt.show()
    #


























def run_mc_temp(temp, args, dic_res, ce):

    string_concentration = args['string_concentration']
    concentration = args['concentration']
    directory_name = args['directory_logs']
    ce_supercell = args['ce_supercell']
    cs = ce.get_cluster_space_copy()
    
    no_steps = args['no_steps']
    SRO_keys = args['SRO_keys']
    interval = args['interval']
    midpoint = args['midpoint']
    
    
    label_supercell = []
    for key in concentration.keys():
        label_supercell += [key]*concentration[key]

    
    np.random.shuffle(label_supercell)
    # label_supercell = [item for sublist in label_supercell for item in sublist]
    ce_supercell_run = ce_supercell.copy()
    ce_supercell_run.set_chemical_symbols(label_supercell)

    calculator = ClusterExpansionCalculator(ce_supercell_run, ce)
    constant_a0 = get_a0(cs.primitive_structure)
    sro_obs = MCBinShortRangeOrderObserver(cs, ce_supercell_run, interval=interval, radius=0.71*constant_a0)
    temp = np.round(temp[0],2)
    filename = '{}/MCce-T{:09.2f}.dc'.format(directory_name, temp)
    mc = CanonicalEnsemble(
        structure=ce_supercell_run,
        calculator=calculator,
        temperature=temp,
        dc_filename=filename,
        ensemble_data_write_interval=interval)
    mc.attach_observer(sro_obs)
    mc.run(number_of_trial_steps = no_steps)
    dc = DataContainer.read(filename)
    steps = dc.data.mctrial.to_numpy()
    dic_res['steps'].append(steps.copy())
    dic_res['structures'].append(dc.data.structure.to_numpy())

    ele_concentrations = sro_obs._get_concentrations(dic_res['structures'][-1][-1])
    dic_res['concentrations'] = ele_concentrations

    good_index = np.where(steps > midpoint)[0]
    
    probs = {}
    for ij in SRO_keys:
        if ele_concentrations[ij[0]]>1E-12 and ele_concentrations[ij[1]] > 1E-12:
            sro_ite = dc.data[(ij[0],ij[1])].to_numpy()
            dic_res[ij].append(sro_ite)
            probs[ij] = (1 - sro_ite[good_index].mean())*(ele_concentrations[ij[0]])*(ele_concentrations[ij[1]])
        else:
            dic_res[ij].append(np.nan)

    ## entropy_calculator
    sum_i = 0
    for i in ele_concentrations.keys():
        if ele_concentrations[i] < 1E-12:
            continue
        sum_i += ele_concentrations[i] * np.log(
            ele_concentrations[i]) - ele_concentrations[i]

    sum_ij = 0
    for i in ele_concentrations.keys():
        for j in ele_concentrations.keys():

            if ele_concentrations[i]>1E-12 and ele_concentrations[j] > 1E-12:
                if (i,j) in probs.keys():
                    sum_ij += probs[(i,j)]*np.log(
                            probs[(i,j)]
                        ) - probs[(i,j)]
                else:
                    sum_ij += probs[(j,i)]*np.log(
                            probs[(j,i)]
                        ) - probs[(j,i)]
    entropy = (12-1)*sum_i - (12/2)*sum_ij + ((12/2) - 1)
    ideal_entropy = 0
    for i in ele_concentrations.keys():
        if ele_concentrations[i] < 1E-12:
            continue
        ideal_entropy += -( ele_concentrations[i] * np.log(
            ele_concentrations[i]) )
    
    obj_function = entropy - 0.95*ideal_entropy
    if obj_function > 0.0:
        obj_function = 20*obj_function
    
    obj_function = np.abs(obj_function)
    
    dic_res['temperature'].append(temp)
    dic_res['entropy'].append(entropy)
    dic_res['ideal_entropy'].append(ideal_entropy)
    dic_res['objective_function'].append(obj_function)
    

    return obj_function







def opt_mc(args, cluster_expansion):
    # try:
    recreate_directory(args['directory_logs'], delete =True)
    log_file_path = args['directory_logs'] + 'log' +'.log'
    log_message('starting opt_mc for {}'.format(args['string_concentration']), log_file_path)
    dic_res = {'steps': []}
    dic_res['args'] = args.copy()
    dic_res['structures'] = []
    dic_res['temperature'] = []
    dic_res['entropy'] = []
    dic_res['ideal_entropy'] = []
    dic_res['objective_function'] = []
    SRO_keys = args['SRO_keys']
    
    for ij in SRO_keys:
        dic_res[ij] = []
    
    X_pre = np.array([600, 1200, 1800, 2400]).reshape(-1, 1)
    # X_pre = np.array([600, 2400]).reshape(-1, 1)
    y = []
    for x_ in X_pre:
        ti = time.time()
        temp = x_
        y_ = run_mc_temp(temp, args, dic_res, cluster_expansion)
        y.append(y_)
        log_message('subsystem {} at {}K took a time of {}s with an objective value of {}'.format(
            args['string_concentration'],temp,time.time() -ti, y_), log_file_path)
    y = np.array(y).reshape(-1,1)
    
    scaler_X = scaler()
    X = scaler_X.fit_transform(X_pre)
    
    kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, nu=1.5)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    plot_GP(gp, X, y, scaler_X, save=True, 
                                            metadata={'subset': args['string_concentration'],
                                            'directory': args['directory_logs'],
                                            'iteration': 0})
    
    plot_entropy(X_pre, dic_res['entropy'], 
                    ideal_entropy = dic_res['ideal_entropy'][-1], 
                    save = True,
                    metadata = {'subset': args['string_concentration'],
                                'directory': args['directory_logs'],
                                'iteration': 0})

    plot_GP_and_entropy(gp, X, y, scaler_X, dic_res['entropy'], ideal_entropy = dic_res['ideal_entropy'][-1], 
                                highlight = 0, save = True, 
                                metadata = {'subset': args['string_concentration'],
                                            'directory': args['directory_logs'],
                                            'iteration': 0})

    log_message('subsystem {}, real values: \n X = {} \n y = {}, scaled values:\n X = {}\n'.format(
        args['string_concentration'], X_pre, y, X), log_file_path)
    
    
    with open(args['directory_logs'] + 'save.pkl', 'wb') as file:
        pickle.dump(dic_res, file)
        
    log_message('GP train and selection start: ', log_file_path)
    for i in range(10):
        x_space = np.linspace(-2.2, 3, 1500).reshape(-1, 1)
    
        ei = gaussian_ei(x_space, gp)
        sorted_space =  np.argsort(ei)[::-1]
        j = 0
        while True:
            x_ = x_space[sorted_space[j]]
            x_ite_pre = scaler_X.inverse_transform([x_])
            if x_ite_pre in X:
                j += 1
            else:
                break
        log_message('subsystem {}, iteration {}, x selected = {} , in non-scaled x = {}'.format(
            args['string_concentration'], i + 1, x_, x_ite_pre), log_file_path)
        y_ = run_mc_temp(x_ite_pre[0], args, dic_res, cluster_expansion)
        
        log_message('subsystem {}, iteration {}, in non-scaled y obtained = {} , '.format(
            args['string_concentration'], i + 1, y_), log_file_path)
        X = np.append(X, x_).reshape(-1,1)
        y = np.append(y, y_).reshape(-1,1)
        
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X, y)
        
        plot_GP(gp, X, y, scaler_X, highlight=1, save=True, 
                                                            metadata={'subset': args['string_concentration'],
                                                            'directory': args['directory_logs'],
                                                                'iteration': i + 1})
        

        X_real = scaler_X.inverse_transform(X)

        plot_entropy(X_real, dic_res['entropy'], 
                    ideal_entropy = dic_res['ideal_entropy'][-1], 
                    save = True,
                    metadata = {'subset': args['string_concentration'],
                                'directory': args['directory_logs'],
                                'iteration': i + 1})

        plot_GP_and_entropy(gp, X, y, scaler_X, dic_res['entropy'], ideal_entropy = dic_res['ideal_entropy'][-1], 
                                highlight = 1, save = True, 
                                metadata = {'subset': args['string_concentration'],
                                            'directory': args['directory_logs'],
                                            'iteration': i + 1})

        dic_res['X_real'] = X_real


        
        dic_res['X'] = X
        dic_res['y'] = y
        
        
        with open(args['directory_logs'] + 'save.pkl', 'wb') as file:
            pickle.dump(dic_res, file)
        

    
    

    plot_entropy(X_real, dic_res['entropy'], 
                    ideal_entropy = dic_res['ideal_entropy'][-1], 
                    save = True,
                    metadata = {'subset': args['string_concentration'],
                                'directory': args['directory_logs']})
    
    with open(args['directory_logs'] + 'save.pkl', 'wb') as file:
        pickle.dump(dic_res, file)
    log_message('Done!', log_file_path)
    # except Exception as e:
    #     print('error: {}'.format(e))
    #     log_message('error: {}'.format(e), log_file_path)

