import random
import time
from pymatgen.io import ase as pg_ase
from datetime import datetime


from .pyiron_calculator import update_failed_job, read_unfinished_job



def write_log(message, log_file='application.log'):
    # Get the current time and format it
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create the full log message
    log_entry = f"{timestamp} - {message}\n"
    
    # Open the log file and append the log entry
    print(log_entry)
    if log_file is not None:
        with open(log_file, 'a') as file:
            file.write(log_entry)


ase_to_pmg = pg_ase.AseAtomsAdaptor.get_structure
pmg_to_ase = pg_ase.AseAtomsAdaptor.get_atoms


def run_packet(df_name, 
                df_global, 
                relaxer_dict,
                input_data=None,
                copy_calculated=None,
                num_cores = 10,
                max_tries = 5,
                **kwargs):
    """why we shouldn't run two of these at the same time?
    if the numbers are repeated they would be looping the same numbers flagging a job node as occupied when 
    two run_packet shouldn't never be run at the same time so the database doesn't handle 
    two different loops saving and loading! (only to o when a changed occured?)
    
    """
    
    calculator_label, relaxer = relaxer_dict['calculator_label'], relaxer_dict['relaxer']
    """ setting up calculator_label  """
    
    df = df_global.copy() # for safekeeping
    
    if calculator_label not in df.columns:
        df[calculator_label] = None
        df[calculator_label] = df[calculator_label].astype(object)
        df.attrs[calculator_label] = {}
        df.attrs[calculator_label]['marked'] = []
        unmarked = df.index
    else:
        unmarked = df.attrs[calculator_label]['unmarked']
    """ setting to_calculate  """
    # copy :
    """
    if copy_calculated is not None:
        if copy_calculated not in df.columns:
            write_log('this relaxer has not been applied yet and cannot be'
                      ' copied therefore',
                      log_file=df_name+'.log')
            return
        if copy_calculated == calculator_label:
            write_log('same relaxer selected for copy,'
                      'please select another relaxer calculated index, or'
                      ' apply a new relaxer',
                      log_file=df_name+'.log')
            return
        uncalculated_size = list(set(uncalculated_size) & 
                                 set(df.attrs[copy_calculated]['calculated']))
    """
    # if list:
    if 'id_list' in kwargs:
        id_list = kwargs['id_list']
        to_calculate = id_list.copy()
    else:
        id_list = None
    """ setting to_calculate  """
    
    # first: packets and sizes

    if 'size_of_packet' in kwargs:
        size_of_packet = kwargs['size_of_packet']
    else:
        size_of_packet = None
    if 'structure_size' in kwargs:
        structure_size = kwargs['structure_size']
    else:
        structure_size = None
    # cont
    
    if size_of_packet is not None and structure_size is not None:
        unmarked_at_size = list(
            set(df[df['size'] == structure_size].index.to_list()) & 
            set(unmarked))
        rest_number = len(unmarked_at_size)
        if rest_number == 0:
            write_log('no more structures to run of this size', 
                      log_file=df_name+'.log')
            return
        if rest_number > size_of_packet:
            to_calculate = random.sample(unmarked_at_size, 
                                         size_of_packet)
        else:
            to_calculate = unmarked_at_size
    
        write_log('attempting to run :\n {}\n'
                'of size :\n {}\n'
                'with calculator :\n {}\n'.format(
                    str(to_calculate), len(to_calculate), calculator_label),
                log_file=df_name+'.log')
    else:
        pass
        

    # instead of running we check
    print('calculating batch {}'.format(to_calculate))
    """we need to start by creating a whole system that calculates 
    the whole indexes 'to_calculate' 
    by starting a pyiron project of name 'relaxer_dict['calculator_label']'
    only when one of them passes state to calculated we added to
    the 'calculated'. We need a higher level symbolism for these labels
    """
    
    busy = 0
    busy_workers = []
    RELAXER = relaxer(project_pyiron = calculator_label, input_data = input_data)
    print('in {} number of cores'.format(num_cores))
    for i, _ in enumerate(to_calculate):
        print('#'*10+'< i: {}, to_calculate[i]: {}  >'.format(i, to_calculate[i])+'#'*10)
        if isinstance(df.loc[to_calculate[i], calculator_label], dict):
            
            # gets master_status from last run
            master_status = df.loc[to_calculate[i], calculator_label]['master_status']
            # options for master status will be FINISHED, RUNNING, TIMEOUT or FAILED
            # master_status is updated within the calculation function
            if  master_status == 'RUNNING' or master_status =='PENDING' or master_status == 'SUBMITTED':
                # update master_status
                df.at[to_calculate[i], calculator_label].update(
                    RELAXER.vasp_pyiron_calculation(id = to_calculate[i])
                )
                master_status = df.loc[to_calculate[i], calculator_label]['master_status']
                print('master_status', master_status)
                if  master_status == 'RUNNING' or master_status =='PENDING' or master_status == 'SUBMITTED':
                    busy += 1
                    busy_workers.append(to_calculate[i])
                    if busy >= num_cores:
                        print('all workers busy, working on {}'.format(busy_workers))
                        break
                    continue # next one in the list, this one busy
            
            if df.loc[to_calculate[i], calculator_label]['master_status'] == 'FAILED':
                # check, update if less than 5 (copy and change status to initiated)
                # else just go to next one
                if 'try_no' in df.loc[to_calculate[i], calculator_label]:
                    if df.loc[to_calculate[i], calculator_label]['try_no'] >= max_tries:
                        print('we tried too many times this structure, last step FAILED')
                        continue
                    df.loc[to_calculate[i], calculator_label]['try_no'] += 1
                    # if try_no already existed we save errors from last
                    try_no = df.loc[to_calculate[i], calculator_label]['try_no']
                else:
                    df.loc[to_calculate[i], calculator_label]['try_no'] = 1
                    try_no = 1

                ### CHECK ######### if already parsed, if not just skip

                if 'error_count' in df.loc[to_calculate[i], calculator_label]:
                    df.loc[to_calculate[i], calculator_label]['error_count'] += 1
                    # if try_no already existed we save errors from last
                    error_count = df.loc[to_calculate[i], calculator_label]['error_count']
                else:
                    df.loc[to_calculate[i], calculator_label]['error_count'] = 1
                    error_count = 1

                df.loc[to_calculate[i], calculator_label] \
                    ['error_{:02d}'.format(try_no - 1)] = \
                        df.loc[to_calculate[i], calculator_label]['error']

                df.at[to_calculate[i], calculator_label]['status_{}'.format(try_no - 1)] = 'FAILED'
                
                df.at[to_calculate[i], calculator_label]['error_count_{}'.format(try_no - 1)] = error_count
                df.at[to_calculate[i], calculator_label]['input_data_{}'.format(try_no - 1)] = \
                    df.at[to_calculate[i], calculator_label]['input_data'].copy()
                df.at[to_calculate[i], calculator_label]['starting_structure_{}'.format(try_no - 1)] = \
                    df.at[to_calculate[i], calculator_label]['starting_structure'].copy()
                
                if try_no < max_tries:
                    print('running again try_no:{}'.format(try_no))
                    """what do we change?
                    structure scale and k points ?"""
                    
                    # first, we copy the aborted job, we do it from a custom function in pyiron_calculator,
                    # so we don't start the project here

                    update_failed_job(calculator_label, to_calculate[i], try_no)
                    
                    # then we just send it again with 
                    org_structure = df.loc[to_calculate[i], 'init_structure'].copy()
                    new_structure = org_structure.copy()
                    new_structure.set_cell(org_structure.cell.array*(1 + error_count*0.025),
                                            scale_atoms=True)
                                            
                    
                    
                    # we update status, how do we keep track of erros?
                    df.at[to_calculate[i], calculator_label].update(
                        RELAXER.vasp_pyiron_calculation(
                        structure = new_structure,
                        id = to_calculate[i],
                        update_data = {'-INCAR-ENCUT': 400 + int(20*error_count)},
                        RE = True,
                        )
                    )
                    busy += 1
                    busy_workers.append(to_calculate[i])
                    if busy >= num_cores:
                        print('all workers busy, working on {}'.format(busy_workers))
                        break
                    continue
                else:
                    print('we tried too many times this structure')
            
            if df.loc[to_calculate[i], calculator_label]['master_status'] == 'TIMEOUT':
                print('this calculation has stopped before its time')
                if 'try_no' in df.loc[to_calculate[i], calculator_label]:
                    if df.loc[to_calculate[i], calculator_label]['try_no'] >= max_tries:
                        print('we tried too many times this structure, last step TIMEOUT')
                        continue
                    df.loc[to_calculate[i], calculator_label]['try_no'] += 1
                    try_no = df.loc[to_calculate[i], calculator_label]['try_no']
                else:
                    df.loc[to_calculate[i], calculator_label]['try_no'] = 1
                    try_no = 1
                
                ### check if already parsed, if not just skip
                # we read unfinished


                df.loc[to_calculate[i], calculator_label].update(
                    read_unfinished_job(calculator_label, to_calculate[i]))
                
                # then we update, if ionic_steps is less than 1 there is no info to update,
                # and we just update the number... _{}

                ionic_steps = df.loc[
                            to_calculate[i], calculator_label]['ionic_steps']
                
                # update
                df.loc[to_calculate[i], calculator_label][
                    'ionic_steps_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['ionic_steps']
                
                df.at[to_calculate[i], calculator_label][
                    'status_{}'.format(try_no - 1)] = 'TIMEOUT'

                if 'error_count' in df.loc[to_calculate[i], calculator_label]:
                    # if try_no already existed we save errors from last
                    error_count = df.loc[to_calculate[i], calculator_label]['error_count']
                else:
                    df.loc[to_calculate[i], calculator_label]['error_count'] = 0
                    error_count = 0

                df.at[to_calculate[i], calculator_label]['error_count_{}'.format(try_no - 1)] = error_count
                df.at[to_calculate[i], calculator_label]['input_data_{}'.format(try_no - 1)] = \
                    df.at[to_calculate[i], calculator_label]['input_data'].copy()
                df.at[to_calculate[i], calculator_label]['starting_structure_{}'.format(try_no - 1)] = \
                    df.at[to_calculate[i], calculator_label]['starting_structure'].copy()




                if ionic_steps < 1:
                    if try_no < max_tries:
                        print('running again try_no:{}'.format(try_no))
                        

                        update_failed_job(calculator_label, to_calculate[i], try_no)
                        
                        # then we just send it again with 
                        org_structure = df.loc[to_calculate[i], 'init_structure'].copy()
                        new_structure = org_structure.copy()
                        new_structure.set_cell(org_structure.cell.array*(1 + error_count*0.025),
                                                scale_atoms=True)
                        # we update status, how do we keep track of erros?
                        df.at[to_calculate[i], calculator_label].update(
                            RELAXER.vasp_pyiron_calculation(
                            structure = new_structure,
                            id = to_calculate[i],
                            update_data = {'-INCAR-ENCUT': 400 + int(20*error_count)},
                            RE = True,
                            )
                        )
                        busy += 1
                        busy_workers.append(to_calculate[i])
                        if busy >= num_cores:
                            print('all workers busy, working on {}'.format(busy_workers))
                            break
                        continue
                    else:
                        print('we tried too many times this structure, last update TIMEOUT, 0 steps')


                    
                else:
                    
                    df.loc[to_calculate[i], calculator_label][
                        'energy_traj_{}'.format(try_no - 1)] = df.loc[
                            to_calculate[i], calculator_label]['energy_traj']
                    df.loc[to_calculate[i], calculator_label][
                        'force_traj_{}'.format(try_no - 1)] = df.loc[
                            to_calculate[i], calculator_label]['force_traj']
                    df.loc[to_calculate[i], calculator_label][
                        'stress_traj_{}'.format(try_no - 1)] = df.loc[
                            to_calculate[i], calculator_label]['stress_traj']
                    df.loc[to_calculate[i], calculator_label][
                        'structures_traj_{}'.format(try_no - 1)] = df.loc[
                            to_calculate[i], calculator_label]['structures_traj']
                    df.loc[to_calculate[i], calculator_label][
                        'energy_electronic_step_{}'.format(try_no - 1)] = df.loc[
                            to_calculate[i], calculator_label]['energy_electronic_step']
                    
                    
                    

                    
                    # first, we copy the aborted job, we do it from a custom function in pyiron_calculator,
                    # so we don't start the project here
                    if try_no < max_tries:
                        print('running again try_no:{}'.format(try_no))
                        update_failed_job(calculator_label, to_calculate[i], try_no)
                        new_structure = df.loc[to_calculate[i], calculator_label][
                            'structures_traj_{}'.format(try_no - 1)][-1].copy()
                        
                        # we update status, how do we keep track of erros?
                        df.at[to_calculate[i], calculator_label].update(
                            RELAXER.vasp_pyiron_calculation(
                            structure = new_structure,
                            id = to_calculate[i],
                            RE = True,
                            )
                        )
                        busy += 1
                        busy_workers.append(to_calculate[i])
                        if busy >= num_cores:
                            print('all workers busy, working on {}'.format(busy_workers))
                            break
                        continue
                    else:
                        print('we tried too many times this structure, last update TIMEOUT')


            if df.loc[to_calculate[i], calculator_label]['master_status'] == 'FINISHED':
                
                if 'try_no' in df.loc[to_calculate[i], calculator_label]:
                    try_no = df.loc[to_calculate[i], calculator_label]['try_no'] + 1
                    if 'status_{}'.format(try_no - 1) in df.at[to_calculate[i], calculator_label].keys():
                        print('this calculation has finished succesfully, already parsed')
                        continue
                else:
                    df.loc[to_calculate[i], calculator_label]['try_no'] = 0
                    try_no = 1
                ### check if already parsed, if yes just skip
                
                print('this calculation has finished succesfully, parsing...')
                if 'error_count' in df.loc[to_calculate[i], calculator_label]:
                    # if try_no already existed we save errors from last
                    error_count = df.loc[to_calculate[i], calculator_label]['error_count']
                else:
                    df.loc[to_calculate[i], calculator_label]['error_count'] = 0
                    error_count = 0

                df.at[to_calculate[i], calculator_label][
                    'status_{}'.format(try_no - 1)] = 'FINISHED'

                df.at[to_calculate[i], calculator_label]['error_count_{}'.format(try_no - 1)] = error_count
                df.at[to_calculate[i], calculator_label]['input_data_{}'.format(try_no - 1)] = \
                    df.at[to_calculate[i], calculator_label]['input_data'].copy()
                df.at[to_calculate[i], calculator_label]['starting_structure_{}'.format(try_no - 1)] = \
                    df.at[to_calculate[i], calculator_label]['starting_structure'].copy()
                    
                df.loc[to_calculate[i], calculator_label][
                    'energy_traj_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['energy_traj']
                df.loc[to_calculate[i], calculator_label][
                    'force_traj_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['force_traj']
                df.loc[to_calculate[i], calculator_label][
                    'stress_traj_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['stress_traj']
                df.loc[to_calculate[i], calculator_label][
                    'structures_traj_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['structures_traj']

                df.loc[to_calculate[i], calculator_label][
                    'ionic_steps_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['ionic_steps']

                df.loc[to_calculate[i], calculator_label][
                    'energy_electronic_step_{}'.format(try_no - 1)] = df.loc[
                        to_calculate[i], calculator_label]['energy_electronic_step']

                
                    

                # check and update
                continue
        else:
            # if the dataframe block is empty, initialize(only the first time)
            
            df.at[to_calculate[i], calculator_label]= RELAXER.vasp_pyiron_calculation(structure = df.loc[
                to_calculate[i], 'init_structure'],
                id = to_calculate[i],
                )
            busy += 1
            busy_workers.append(to_calculate[i])
            if busy >= num_cores:
                print('all workers busy, working on {}'.format(busy_workers))
                break
    if len(busy_workers) < num_cores:
        print('current workers working on {}'.format(busy_workers))
    df.attrs[calculator_label]['unmarked'] = list(
        set(unmarked) - set(to_calculate))

    df.attrs[calculator_label]['marked'] += to_calculate
    df.attrs[calculator_label]['marked'] = list(
        set(df.attrs[calculator_label]['marked']))
    
    
    
    df.to_pickle(df_name+'.pkl')

    return df

    
# else: 
        #     df.at[to_calculate[i], calculator_label]= {'status':'not started'}
        #     print('not started/not info on {:015d}'.format(to_calculate[i]))
        #     print('starting')

            
        
  # obj.attr_name exists.df.loc[to_calculate[i], calculator_label].)
        



    # ti = time.time()
    # df.loc[to_calculate, calculator_label] = df.loc[to_calculate].apply(
    #     relaxer, log_file=df_name+'.log', axis=1)
    # ti = time.time() - ti
    