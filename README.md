# disall
Disordered alloy working parallely within the HPRC system

How to use in FASTER

FASTER has access to VASP, and an scheduling system that allows to run different structures parallely.
FASTER also has access to great RAM capbilities needed to run batches of structures.
object directives such as grid_gen, and grid_run are expected to run in a SLUR manager machine.

Launch has better access to GPUs and can only be accessed through the portal, therefore we use it to train and test MLAIPs. Including hyperparameter optimization.
