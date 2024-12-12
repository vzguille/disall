import os
from pathlib import Path
import uuid
import subprocess

from ase.io.espresso import write_espresso_in
from ase.io.espresso import read_espresso_out

from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer
from disalloy.grid_run import write_log

from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms

PSEUDOPOTENTIALS = {
    'Al':'Al.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Co':'Co_pbe_v1.2.uspp.F.UPF',
    'Cr':'cr_pbe_v1.5.uspp.F.UPF',
    'Cu':'Cu.paw.z_11.ld1.psl.v1.0.0-low.upf',
    #'Cu':'Cu.pbe-kjpaw.UPF',
    'Fe':'Fe.pbe-spn-kjpaw_psl.0.2.1.UPF',
    'Mn':'mn_pbe_v1.5.uspp.F.UPF',
    'Ni':'ni_pbe_v1.4.uspp.F.UPF',
    'V':'v_pbe_v1.4.uspp.F.UPF',
     }


INPUT_DATA = {
    'pseudo_dir':'/usr/src/SSSP_efficiency/',
    'system': {'ecutwfc': 60, 'ecutrho': 480},
    'disk_io': 'low',  # Automatically put into the 'control' section
    'tstress':True,  # deprecated, put in input_data
    'tprnfor':True,  # deprecated, put in input_data
    'calculation': 'vc-relax',#'scf',
    'conv_thr':1e-8,
    'occupations':'smearing',
    'smearing':'mp',
    'degauss':0.015,
    'mixing_beta':0.3,
    #'mixing_ndim':10,
    'electron_maxstep':200,

    # 'nspin':2
    'diagonalization':'cg',
}

def parse_calculation_traj_relax(file_name = 'espresso.pwo'):
    res = []
    with open(file_name, 'r') as f:
        for i in read_espresso_out(f, results_required=True):
            res.append([i.copy(), i.get_potential_energy(), i.get_forces(), i.get_stress()])
    return res

class PW_calculator:
    def __init__(self,
                input_data = INPUT_DATA,
                pseudopotentials = PSEUDOPOTENTIALS):
        self.input_data = input_data
        self.pseudopotentials = pseudopotentials
        



    def qe_calculation(self,
                    qe_structure, #calculation specific
                    qe_id = 0, #calculation specific
                    log_file = None, #calculation specific
                    np=1, # calculation_specific
                    **kwargs):
        if 'input_data' in kwargs:
            input_data = kwargs['input_data']
        else:
            input_data = self.input_data
        if 'pseudopotentials' in kwargs:
            pseudopotentials = kwargs['pseudopotentials']
        else:
            pseudopotentials = self.pseudopotentials
        
        folder_name = 'calculation_{:08d}_{}'.format(qe_id,uuid.uuid4())
        Path(folder_name).mkdir(parents=True, exist_ok=True)

        obj_magmoms = CollinearMagneticStructureAnalyzer(
        ase_to_pmg(qe_structure),
        'replace_all')
        qe_structure.set_initial_magnetic_moments(
                    obj_magmoms.magmoms)
        
        write_espresso_in(os.path.join(folder_name, 'pw.in'), 
                        qe_structure, 
                        input_data=input_data,
                        pseudopotentials=pseudopotentials,
                        kspacing=0.5e-1,
                        format='espresso-in')
        command = ["mpirun", "-np", str(np), "-allow-run-as-root", 
                    "/usr/qe/bin/pw.x"]
        
        # Run the binary inside the unique folder
        with open(os.path.join(folder_name, 'pw.in'), "r") as infile, \
        open(os.path.join(folder_name, 'pw.out'), "w") as outfile, \
        open(os.path.join(folder_name, 'pw.err'), "w") as errfile:
            result = subprocess.run(command, 
                                    stdin=infile, 
                                    stdout=outfile,
                                    stderr=errfile,
                                    cwd=folder_name,
                                    text=True)
        if result.returncode != 0:
            write_log(f"Error occurred: {result.stderr}",
            log_file=log_file)
        else:
            write_log(f"Calculation {qe_id:08d} completed in folder {folder_name}",
            log_file=log_file)
            return parse_calculation_traj_relax(os.path.join(folder_name, 'pw.out'))