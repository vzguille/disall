import random
import time
from pymatgen.io import ase as pg_ase
from datetime import datetime

from .pyiron_calculator import update_failed_job


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
                
                copy_calculated=None,
                num_cores = 10,
                **kwargs):
    """why we shouldn't run two of these at the same time?
    if the numbers are repeated they would be looping the same numbers flagging a job node as occupied when 
    two run_packet shouldn't never be run at the same time so the database doesn't handle 
    two different loops saving and loading! (only to o when a changed occured?)
    
    """
    calculator_label, relaxer = relaxer_dict['calculator_label'], relaxer_dict['relaxer']
    """ setting up calculator_label  """
    df = df_global
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
    print(to_calculate)
    """we need to start by creating a whole system that calculates 
    the whole indexes 'to_calculate' 
    by starting a pyiron project of name 'relaxer_dict['calculator_label']'
    only when one of them passes state to calculated we added to
    the 'calculated'. We need a higher level symbolism for these labels
    """
    
    busy = 0
    busy_running = []
    RELAXER = relaxer(project_pyiron = calculator_label)
    print(num_cores)
    for i, _ in enumerate(to_calculate):
        print('i: {}, to_calculate[i]: {}'.format(i, to_calculate[i]))
        if isinstance(df.loc[to_calculate[i], calculator_label], dict):
            status = df.loc[to_calculate[i], calculator_label]['status']
            if  status == 'initiated' or status == 'running' or status == 'submitted':
                
                busy_running.append(to_calculate[i])
                result_ite = RELAXER.vasp_pyiron_calculation(id = to_calculate[i])
                df.at[to_calculate[i], calculator_label].update(result_ite)
                status = df.loc[to_calculate[i], calculator_label]['status']
                if  status == 'initiated' or status == 'running' or status == 'submitted':
                    busy += 1
                    if busy >= num_cores:
                        break
                    continue # next one in the list, this one busy
                else:
                    pass # it less the iteration to go on

            if df.loc[to_calculate[i], calculator_label]['status'] == 'failed':
                # check, update if less than 5 (copy and change status to initiated)
                # else just go to next one
                continue
            if df.loc[to_calculate[i], calculator_label]['status'] == 'finished':
                print('this calculation has finished succesfully')
                # check and update
                continue
            if df.loc[to_calculate[i], calculator_label]['status'] == 'aborted':
                print('this calculation has been aborted')
                if 'try_number' in df.loc[to_calculate[i], calculator_label]:
                    df.loc[to_calculate[i], calculator_label]['try_number']+=1
                else:
                    df.loc[to_calculate[i], calculator_label]['try_number']=1
                try_no = df.loc[to_calculate[i], calculator_label]['try_number']
                
                if try_no < 5:
                    print('running again')
                    """what do we change?
                    structure scale and k points ?"""
                    
                    # first, we copy the aborted job, we do it from a custom function in pyiron_calculator,
                    # so we don't start the project here

                    update_failed_job(calculator_label, to_calculate[i], try_no)
                    
                    # then we just send it again with 
                    org_structure = df.loc[to_calculate[i], 'init_structure'].copy()
                    new_structure = org_structure.copy()
                    new_structure.set_cell(org_structure.cell.array*(1 + try_no*0.025),
                                            scale_atoms=True)
                                            
                    RELAXER.vasp_pyiron_calculation(
                        structure = new_structure,
                        id = to_calculate[i],
                        update_data = {'-INCAR-ENCUT': 400 + int(20*try_no)},
                        RE = True,
                        )
                    
                    # we update status, how do we keep track of erros?
                    df.at[to_calculate[i], calculator_label]['status'] = 'running'
                    continue
                else:
                    print('we tried too many times this structure')
                
                
        else:
            RELAXER.vasp_pyiron_calculation(structure = df.loc[to_calculate[i], 'init_structure'],
                                                         id = to_calculate[i],
                                                         )
            df.at[to_calculate[i], calculator_label]= {'status':'running'}
            busy += 1
            if busy >= num_cores:
                break
    
    df.attrs[calculator_label]['unmarked'] = list(
        set(unmarked) - set(to_calculate))

    df.attrs[calculator_label]['marked'] += to_calculate
    df.attrs[calculator_label]['marked'] = list(
        set(df.attrs[calculator_label]['marked']))
    

    df.to_pickle(df_name+'.pkl')

    
# else: 
        #     df.at[to_calculate[i], calculator_label]= {'status':'not started'}
        #     print('not started/not info on {:015d}'.format(to_calculate[i]))
        #     print('starting')

            
        
  # obj.attr_name exists.df.loc[to_calculate[i], calculator_label].)
        



    # ti = time.time()
    # df.loc[to_calculate, calculator_label] = df.loc[to_calculate].apply(
    #     relaxer, log_file=df_name+'.log', axis=1)
    # ti = time.time() - ti
    