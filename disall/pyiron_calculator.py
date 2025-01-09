import os
from pyiron import Project
from pyiron import ase_to_pyiron

from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io import ase as pgase

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
                input_data = INPUT_DATA,
                project_pyiron = 'testing_project',
                
                ):
        self.dic_parameters = input_data
        self.project_pyiron_str = project_pyiron

    def vasp_pyiron_calculation(self,
                    structure = None, #calculation specific, only when you initiate it uses it
                    id = 0, #calculation specific
                    log_file = None, #calculation specific
                    RE = False, # Don't delete job unless specified
                    **kwargs):
        if 'input_data' in kwargs:
            """ we can also change it in case we want to have given
            VASP parameters in this specific structure """
            dic_parameters = kwargs['input_data']
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
        print(job_function.status)

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
            return

        if job_function.status == 'running' or \
                job_function.status == 'collect' or \
                job_function.status == 'submitted':
            return {'status' : job_function.status.string}

        if job_function.status == 'finished' or job_function.status == 'aborted' \
                or job_function.status == 'not_converged' or \
                job_function.status == 'warning':
            try:
                """ when it succeds parsing the results"""
                return _get_dft_results_pyiron(job_function)
            except Exception as e:
                """ when it fails it returns a False (this is important for the standalone relaxer),
                any other automatic updating has to be done outside in the loop
                """
                print(e)
                return {'status': job_function.status.string, 'error': e}

                #     err_count = self._fail(job_function, it_in_rooster, e)
                #     if err_count >= 5:
                #         print('giving up on this run')
                #     else:
                #         self._fail_update(
                #             jobs, i, it_in_rooster, proj_pyiron, err_count)
                #         continue
                # finished worked
            return
def _relax_blank(job, dic_parameters):
    for key, value in dic_parameters.items():
        if key.startswith('-INCAR-'):
            job.input.incar[key.split('-INCAR-')[-1]] = value
    
    if '-INCAR-ISPIN' in dic_parameters.keys():
        if dic_parameters['-INCAR-ISPIN'] == 2:
            # if 'Fe',' Ni' in job.structure set ISPIN 1 and break
            obj_magmoms = CollinearMagneticStructureAnalyzer(ase_to_pmg(job.structure),
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
        job.server.queue = job.server.list_queues()[1]
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
        work_str = Kpoints().automatic_density(pgase.AseAtomsAdaptor.get_structure(
            job.structure), kppa=KPPA)
        
        print('KPOINTS file')
        print(work_str)
        
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

def _get_dft_results_pyiron(job_func):
    return {'job_name': job_func.job_name,
            'status': job_func.status.string,
            'energy_traj': job_func['output/generic/energy_pot'].copy(),
            'force_traj': job_func['output/generic/forces'].copy(),
            'stress_traj': job_func['output/generic/stresses'].copy(),
            
            'structures_traj': [
                j.to_ase().copy() for j in job_func.trajectory()],
            'energy_electronic_step': job_func[
                'output/generic/dft/scf_energy_int'].copy()}



def update_failed_job(project_name='testing_project', id = 0, try_no = 0):
    
    proj_pyiron = Project(path=project_name)
    job = proj_pyiron.load('vasp_{:015d}'.format(
                                    id
                                    ))
    job.copy_to(new_job_name=job.name + '_err_{:02}'.format(try_no - 1), 
                    copy_files=True, 
                    new_database_entry=False)