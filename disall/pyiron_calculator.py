import os
import re

import shutil
import time

import numpy as np
from pyiron import Project
from pyiron import ase_to_pyiron

from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.io.vasp.inputs import Kpoints


from pymatgen.io.vasp import Vasprun

from .slurm import squeue_check_av

from pymatgen.io import ase



ase_to_pmg = ase.AseAtomsAdaptor.get_structure
pmg_to_ase = ase.AseAtomsAdaptor.get_atoms

"""how to call the calculator
    pw_calc = PW_calculator()
    pw_calc.qe_calculation(atoms_test, id_test, np=4) 
    Therefore calculation needs to be self sufficient:
    * needs to have standard parameters
    * if it doesn't exist
        * check if there isn't a zip - > run
    * if it exists, checks and recollect

each relaxation technique has to be     
"""

INPUT_DATA = {'-INCAR-NCORE': 16,
                            '-INCAR-LORBIT': 1,
                            '-INCAR-NSW': 200,
                            '-INCAR-PREC': 'Accurate',
                            '-INCAR-IBRION': 2,
                            '-INCAR-ISMEAR': 0,
                            '-INCAR-SIGMA': 0.03,
                            '-INCAR-ENCUT': 400,
                            '-INCAR-EDIFF': 1E-6,
                            '-INCAR-ISIF': 2,
                            '-INCAR-ISYM': 0,
                            'KPPA': 3000,
                            '-INCAR-ISPIN': 1,
                            }

class VASP_pyiron_calculator:
    def __init__(self,
                project_pyiron = 'testing_project',
                **kwargs,
                ):
        
        self.project_pyiron_str = project_pyiron
        if 'input_data' in kwargs:
            """ we can also change it in case we want to have given
            VASP parameters in this specific structure """
            self.dic_parameters = kwargs['input_data']
            if kwargs['input_data'] is None:
                self.dic_parameters = INPUT_DATA
        else:
            self.dic_parameters = INPUT_DATA

    def vasp_pyiron_calculation(self,
                    structure = None, #calculation specific, only when you initiate it uses it
                    id = 0, #calculation specific
                    log_file = None, #calculation specific
                    RE = False, # Don't delete job unless specified,
                    return_job = False,
                    **kwargs):
        if 'input_data' in kwargs:
            """ we can also change it in case we want to have given
            VASP parameters in this specific structure """
            dic_parameters = kwargs['input_data']
            if kwargs['input_data'] is None:
                dic_parameters = self.dic_parameters
        else:
            dic_parameters = self.dic_parameters

        if 'update_data' in kwargs:
            update_data = kwargs['update_data']
            dic_parameters.update(update_data)
        else: 
            pass

        if 'project_name' in kwargs:
            project_name = kwargs['project_name']
        else: 
            project_name = self.project_pyiron_str
        
        proj_pyiron = Project(path=project_name)
        
        try:
            job_function = proj_pyiron.create_job(
                job_type=proj_pyiron.job_type.Vasp,
                job_name='vasp_{:015d}'.format(
                                    id
                                    ),
                delete_existing_job=RE)
                
        except Exception as e:
            print('Exception in loading job: {}'.format(e))
            job_function = proj_pyiron.load('vasp_{:015d}'.format(
                                    id
                                    ))
        print('pyiron_status start of loop: {}'.format(job_function.status))

        """depending on pyiron status"""

        if job_function.status == 'initialized':
            job_function.structure = ase_to_pyiron(structure.copy())
            # job function change
            print('PASS to RUN --> id{:015d}'.format(
                    id
                    ))
            # actually run it
            _relax_blank(
                job_function, dic_parameters)
            if return_job:
                return job_function
            return {'master_status' : 'SUBMITTED',
                    'job_name': job_function.job_name,
                    'QID': job_function['server']['qid'],
                    'error': '',
                    'status_pyiron' : job_function.status.string,
                    'input_data': dic_parameters,
                    'starting_structure':structure.copy(),
                    'energy_traj': None, # in eV
                    'force_traj': None, # in eV/A
                    'stress_traj': None, # in eV/A3, MatGL uses GPa
                    
                    'structures_traj': None,
                    'energy_electronic_step': None,
                    'ionic_steps': None
                    }

        if job_function.status == 'running' or job_function.status == 'submitted': # or job_function.status == 'collect'
            #check if it's actually running
            print('pyiron QID', job_function['server']['qid'])
            squeue_valid = False
            for i in range(3):
                try:
                    status_slurm, run_time = squeue_check_av(job_function['server']['qid'])
                    QID = job_function['server']['qid']
                    squeue_valid = True
                    break
                except Exception as e:
                    print('SQUEUE ERROR {}'.format(i))
                    print(e)
                    time.sleep(0.2)
            if not squeue_valid:
                print('running/submitted alternate squeue check, status: {}'.format(
                    job_function.status))
                QID = _find_job_number(job_function.working_directory)
                status_slurm, run_time = squeue_check_av(QID)
                print('alternate result, QID: {}, status_slurm: {}, run_time: {}'.format(
                    QID, status_slurm, run_time
                ))
            print('status_pyiron: {}, status_slurm: {}'.format(job_function.status, status_slurm))
            # if not squeue_valid:
            #     job_function.drop_status_to_aborted() # error in queues, we just give up
            #     return {'master_status' : 'FAILED',
            #         'job_name': job_function.job_name,
            #         'QID': job_function['server']['qid'],
            #         'error': 'queue failed',
            #         'status_pyiron' : job_function.status.string,
            #         'input_data': dic_parameters,
            #         'starting_structure': structure.copy(),
            #         'energy_traj': None, # in eV
            #         'force_traj': None, # in eV/A
            #         'stress_traj': None, # in eV/A3, MatGL uses GPa
            #         'structures_traj': None,
            #         'energy_electronic_step': None,
            #         'ionic_steps': None
            #         }
            if status_slurm == 'PENDING' or status_slurm == 'RUNNING' or status_slurm == 'COMPLETING':
                master_status = status_slurm
                if return_job:
                    return job_function
                return {'master_status' : status_slurm,
                        'status_pyiron' : job_function.status.string}
            else:
                job_function.drop_status_to_aborted()
                print('dropping pyiron status to aborted')
                master_status = 'TIMEOUT'
                if return_job:
                    return job_function
                
                

        if job_function.status == 'finished' \
            or job_function.status == 'aborted' \
            or job_function.status == 'not_converged' \
            or job_function.status == 'warning':
            squeue_valid = False
            print('pyiron QID', job_function['server']['qid'])
            for i in range(3):
                try:
                    status_slurm, run_time = squeue_check_av(job_function['server']['qid'])
                    QID = job_function['server']['qid']
                    squeue_valid = True
                    break
                except Exception as e:
                    print('SQUEUE ERROR {}'.format(i))
                    print(e)
                    time.sleep(0.2)
            if not squeue_valid:
                
                print('finished/aborted alternate squeue check, status: {}'.format(
                    job_function.status))
                # run_queue error returns an error here until it gets a QID,
                # therefore, calc returns None
                QID = _find_job_number(job_function.working_directory)
                status_slurm, run_time = squeue_check_av(QID)
                print('alternate result, QID: {}, status_slurm: {}, run_time: {}'.format(
                    QID, status_slurm, run_time
                ))
            
            print('status_pyiron: {}, status_slurm: {}'.format(job_function.status, status_slurm))

                        
            if os.path.exists(job_function.working_directory+'/truncated_vasprun.xml'):
                master_status = 'TIMEOUT'
                print('reverting aborted to TIMEOUT due to previous error')
            if status_slurm == 'RUNNING':
                master_status = 'RUNNING'
                return {'master_status' : 'RUNNING',
                            'status_pyiron': job_function.status.string,
                            'error': '',
                            'QID': QID, 
                            'run_time': run_time,
                            }
            try:
                print('parsing pyiron job...')
                """ when it succeds parsing the results"""
                if return_job:
                    return _get_dft_results_pyiron(job_function, return_job = return_job)
                else:
                    
                    res_dir = _get_dft_results_pyiron(job_function)
                    res_dir.update({'QID': QID, 'run_time': run_time})
                    return res_dir
                time.sleep(0.5)
            except Exception as e:
                """ when it fails it returns a False (this is important for the standalone relaxer),
                any other automatic updating has to be done outside in the loop
                """
                print('parsing not succesful')
                if return_job:
                    return job_function
                if 'master_status' in locals():
                    if master_status == 'TIMEOUT':
                        print('TIMEOUT')
                        print(e)
                        ### parse it when it's run_timed out
                        return {'master_status' : 'TIMEOUT',
                            'status_pyiron': job_function.status.string,
                            'error': e,
                            'QID': QID, 
                            'run_time': run_time,
                            }
                else:
                    print('FAILED')
                    print(e)
                    return {'master_status':'FAILED',
                        'status_pyiron': job_function.status.string,
                        'error': e,
                        'QID': QID, 
                        'run_time': run_time,
                        }



def _relax_blank(job, dic_parameters):
    for key, value in dic_parameters.items():
        if key.startswith('-INCAR-'):
            job.input.incar[key.split('-INCAR-')[-1]] = value
    
    if '-INCAR-ISPIN' in dic_parameters.keys():
        if dic_parameters['-INCAR-ISPIN'] == 2:
            # if 'Fe',' Ni' in job.structure set ISPIN 1 and break
            obj_magmoms = CollinearMagneticStructureAnalyzer(
                ase_to_pmg(job.structure),
                'replace_all')
            job.structure.set_initial_magnetic_moments(
                obj_magmoms.magmoms)
    
    if 'KPPA' in dic_parameters.keys():
        KPPA = dic_parameters['KPPA']
    else:
        KPPA = 3000

    if 'queue_number' in dic_parameters.keys():
        job.server.queue = job.server.list_queues()[dic_parameters['queue_number']]
    else:
        job.server.queue = job.server.list_queues()[0]

    if len(job.structure) == 1:
        job.server.queue = job.server.list_queues()[0]
        job.input.incar['NCORE'] = 1
    
    elif len(job.structure) > 1 and len(job.structure) <= 5:
        job.server.queue = job.server.list_queues()[len(job.structure) - 1]

    elif len(job.structure) > 5 and len(job.structure) <= 10:
        job.server.queue = job.server.list_queues()[2]
    elif len(job.structure) > 10 and len(job.structure) <= 15:
        job.server.queue = job.server.list_queues()[3]
    elif len(job.structure) > 15 and len(job.structure) <= 20:
        job.server.queue = job.server.list_queues()[4]
    elif len(job.structure) > 20 and len(job.structure) <= 25:
        job.server.queue = job.server.list_queues()[5]
    elif len(job.structure) > 25 and len(job.structure) <= 30:
        job.server.queue = job.server.list_queues()[6]
    elif len(job.structure) > 30:
        job.server.queue = job.server.list_queues()[7]
    
    


    if 'k_mesh' in dic_parameters.keys():
        if dic_parameters['k_mesh'] is list:
            job.k_mesh = dic_parameters['k_mesh']
        else:
            job.k_mesh_spacing = dic_parameters['k_mesh']
    else:
        work_str = Kpoints().automatic_density(ase_to_pmg(
            job.structure), kppa=KPPA)
        
        print('SENDING...')
        time.sleep(2)
        # print('KPOINTS file')
        # print(work_str)
        
        with open('KPOINTS', 'w') as file:
            file.write(str(work_str))
        job.copy_file_to_working_directory('KPOINTS')

        # job.set_kpoints(mesh=work_str.as_dic_parameterst()[
        #    'kpoints'][0], center_shift=[0, 0, 0])
        # job.set_kpoints_file(method='Gamma', 
        #    size_of_mesh=mesh=work_str.as_dic_parameterst()['kpoints'][0])
    job.run()
    if os.path.exists('KPOINTS'):
        os.remove('KPOINTS')
    

def _get_dft_results_pyiron(job_func, return_job = False):
    CONVERSION_to_ = 1602.176621 # transforming back to VASP values 
    if return_job:
        return job_func
    else:
        status_pyiron = job_func.status.string
        return {'master_status': 'FINISHED',
                'error':'',
                
                'status_pyiron': job_func.status.string,
                
                'energy_traj': job_func['output/generic/dft/energy_zero'].copy(), # in eV
                'force_traj': job_func['output/generic/forces'].copy(), # in eV/A
                'stress_traj': job_func['output/generic/stresses'].copy()*CONVERSION_to_, # in eV/A3, MatGL uses GPa
                
                'structures_traj': [
                    j.to_ase().copy() for j in job_func.trajectory()],
                'energy_electronic_step': job_func[
                    'output/generic/dft/scf_energy_zero'].copy(),
                'ionic_steps': len(job_func['output/generic/dft/energy_zero'])}



def update_failed_job(project_name = 'testing_project', id = 0, try_no = 0):
    
    proj_pyiron = Project(path=project_name)
    job = proj_pyiron.load('vasp_{:015d}'.format(
                                    id
                                    ))
    job.copy_to(new_job_name=job.name + '_err_{:02}'.format(try_no - 1), 
                    copy_files=True, 
                    new_database_entry=False)
    
    # we leave it for later                
    # _zip_job_dir(project_name +'/'+ job.name + '_err_{:02}'.format(try_no - 1))

def _zip_job_dir(dir_name, DEL_AFTER=True):
        _zip_archive(dir_name,
                     dir_name + '.zip')
        if DEL_AFTER:
            shutil.rmtree(dir_name)
            
def _zip_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s' % (name, format), destination)


def read_unfinished_job(project_name = 'testing_project', id = 0, vasprun_file = None):
    print('reading unfinished...')
    
    
    
    

    if vasprun_file is not None:
        file_path = vasprun_file
        working_directory = '/'.join(vasprun_file.split('/')[:-1])
    else:
        proj_pyiron = Project(path=project_name)
        
        
        job = proj_pyiron.load('vasp_{:015d}'.format(
                                        id
                                        ))
        
        squeue_valid = False
        for i in range(5):
            try:
                status_slurm, run_time = squeue_check_av(job['server']['qid'])
                squeue_valid = True
                break
            except Exception as e:
                print('SQUEUE ERROR {}'.format(i))
                print(e)
                time.sleep(1)
        working_directory = job.working_directory                               
        # Open the original file for reading
        file_path = job.working_directory+'/vasprun.xml'
    keyword = '</calculation>'
    additional_line = '</modeling>'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            last_occurrence = -1
            for i, line in enumerate(lines):
                if keyword in line:
                    last_occurrence = i
    else:
        print('vasprun.xml NOT CREATED')
        last_occurrence = -1

    if last_occurrence == -1:
        print('NOT ONE step culminated')
        return {'ionic_steps': 0,
        }
    # If the keyword is found, keep lines up to and including the last occurrence
    if last_occurrence != -1:
        truncated_lines = lines[:last_occurrence + 1]
    else:
        truncated_lines = lines

    # Add the additional line at the end
    truncated_lines.append(additional_line + '\n')

    # Write the truncated lines to the new file
    with open(working_directory+'/truncated_vasprun.xml', 'w') as file:
        file.writelines(truncated_lines)
    vasprun = Vasprun(working_directory+'/truncated_vasprun.xml')
    energy_traj = []
    force_traj = []
    stress_traj = []
    structures_traj = []
    energy_electronic_step = []

    
    for i in vasprun.ionic_steps:
        energy_traj.append(i['e_0_energy'])
        force_traj.append(i['forces'])
        stress_traj.append(i['stress'])
        structures_traj.append(i['structure'].to_ase_atoms().copy())
        #structures_traj.append(pmg_to_ase())
        energy_electronic_step.append([])
        for j in i['electronic_steps']:
            energy_electronic_step[-1].append(j['e_0_energy'])
        energy_electronic_step[-1] = np.array(energy_electronic_step[-1])
    
    
    energy_traj = np.array(energy_traj)
    force_traj = np.array(force_traj)
    stress_traj = np.array(stress_traj)
    energy_electronic_step = np.array(energy_electronic_step, dtype= object)
        

    print('{} steps culminated'.format(len(energy_traj)))
    return {'run_time': run_time,
                'energy_traj': energy_traj, # in eV
                'force_traj': force_traj, # in eV/A
                'stress_traj': stress_traj, # in GPa
                'structures_traj': structures_traj,
                'energy_electronic_step': energy_electronic_step,
                'ionic_steps': len(energy_traj)}


        # print('e_0_energy', i['e_0_energy'])
        # print('forces', i['forces'])
        # print('stress', i['stress'])
        # print('structure', i['structure'].to_ase_atoms().copy())
        # print(dir(i))
    # trajectory = vasprun.get_trajectory()
    # if len(trajectory) > 0:
    #     structure = trajectory[-1]
    # else: 
    #     structure = trajectory[0]
    # # print(dir(structure))
    # new_structure = structure.to_ase_atoms()

    # job.copy_to(new_job_name=job.name + '_err_{:02}'.format(try_no - 1), 
    #     copy_files=True, 
    #     new_database_entry=False)
    
# Define the file paths
def _find_job_number(directory):
    for filename in os.listdir(directory):
        match = re.match(r"job\.l(\d+)", filename)
        if match:
            return match.group(1)  # Extract the number
    return None  # Return None if no matching file is found