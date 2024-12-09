import numpy as np 
import time
from itertools import groupby, combinations



def find_ensembles(n, m, current_ensemble=[], start=1):
    """
    Find all unique ensembles of integer numbers that sum up to n with a maximum length of m.

    Args:
    - n: the target sum
    - m: the maximum length of the ensembles
    - current_ensemble: the current ensemble being built
    - start: the minimum value to start adding to the current ensemble

    Returns:
    - A list of lists, where each list is a unique ensemble of integers that sum up to n and has a maximum length of m.
    """
    ensembles = []

    # Base case: If n is 0 and the current ensemble has length m, it's a valid solution.
    if n == 0 and len(current_ensemble) <= m:
        ensembles.append(current_ensemble.copy())
        return ensembles

    # Recursive step: Try adding numbers from start to n to the current ensemble, with a maximum length of m.
    for i in range(start, n + 1):
        current_ensemble.append(i)
        # Recursively find ensembles that sum up to n - i, with a maximum length of m, starting from i.
        ensembles.extend(find_ensembles(n - i, m, current_ensemble, i))
        current_ensemble.pop()  # Backtrack

    return ensembles

def unique_permutations(nums):
    if len(nums) == 1:
        return [nums]

    result = []
    for i, num in enumerate(nums):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # Skip duplicate elements
        remaining_nums = nums[:i] + nums[i + 1:]
        perms = unique_permutations(remaining_nums)
        for perm in perms:
            result.append([num] + perm)

    return result


def separate_by_length(list_of_lists):
    list_of_lists.sort(key=len)  # Sort the list of lists by length
    return {k: list(g) for k, g in groupby(list_of_lists, key=len)}




def get_integer_compositions(n, s, minsub, maxsub):
	"""
	Args:
	- n: divisions in grid (10 -> steps of 0.1)
	- s: size of the system
	- minsub : the maximum length of subsystem
	- maxsub : the maximum length of subsystem

	Returns:
	- An array of all the unique permutations of in which s distinct labels
	can fill a container of size n when the minimum and maximum number of distinct
	labels are minsub and maxsub respectively
	"""
	final_comp = np.empty([0, s], int)

	# if maxsub > s:
	# 	print('WARNING maximum subsystem length is larger than the system\'s length''\n'
	# 		'Printing up to the system\'s length {}'.format(s) )
	
	ensembles  = find_ensembles(n, maxsub)
	ensembles = separate_by_length(ensembles)
	

	for i in range(minsub, min(maxsub, n, s) + 1):
		# min here means we cannot create subsystems larger than our system,
		# we also cannot fit a larger than n label subsystem into an smaller 
		# grid division

		perms_per_size = []
		for j in ensembles[i]:
			perms_per_size.extend(unique_permutations(j))

		for comb in combinations(range(s), i):
			dummy = np.zeros([len(perms_per_size), s], dtype=int)
			dummy[:, comb] = perms_per_size
			final_comp = np.vstack((final_comp, dummy))

	return final_comp

def generate_composition_simplex_grid(step_size, system_size, 
	min_subsystem_size, max_subsystem_size):

	return (1/step_size)*get_integer_compositions(step_size, system_size, 
		min_subsystem_size, max_subsystem_size)
