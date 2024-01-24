import numpy as np
import pandas as pd
import uuid
import copy

class bb_queue():

    def __init__(self) -> None:
        self.queue = []

    def add(self,search_regions):
        for region in search_regions:
            self.queue.append(region)

    def prune(self,max_lower_bound):
        pass

    def pop(self):
        candidate_region_not_found = False

        while candidate_region_not_found:
            candidate = self.queue.pop
            if candidate.upper_bound > max_lower_bound:
                candidate_region_not_found

        return candidate

class sigmoidal_function:

    def __init__(self,mu,lam,theta):
        self.mu = mu
        self.lam = lam
        self.theta = theta

    def evaluate(self,x):
        pass

    def get_envelope(self):
        pass

class convex_envelope:
    def __init__(self) -> None:
        pass

def solve_convex_program():
    #newton step and barrier method
    pass

def find_2_line_intersection(line1,line2):
    a0,b0 = line1
    a1,b1 = line2

    return (b0 - ((-a1*b0)/a0 + b1)/(1-(a1/a0)))/-a0

def is_vertex_feasible(vertex,constraints):

    for constr in constraints:
        pass
        # if constr['dir']


def init_search_region(constraints):
    #need max and min feasible vals for each coordinate
    #since the set is closed, need to find vertices (and if nonlinearities, crit points)
    #if dec vars are all positive,
    if is_vertex_feasible(vertex,constraints):
        if vertex_curr_coord_val > max_curr_coord_val:
            max_curr_coord_val = vertex_curr_coord_val

        elif vertex_curr_coord_val < min_curr_coord_val:
            min_curr_coord_val = vertex_curr_coord_val


    pass

class search_region:

    def __init__(self,dims:int,upper_lims:[],lower_lims:[]):
        self.dims = dims
        self.upper_lims = upper_lims
        self.lower_lims = lower_lims
        self.constraints = {}

    def construct_constraints(self):

        self.constraints = {}

        for decision_var in range(self.dims):
            constraint_id = str(uuid.uuid4())
            self.constraints[constraint_id]['var'] = decision_var
            self.constraints[constraint_id]['val'] = self.upper_lims[decision_var]
            self.constraints[constraint_id]['dir'] = 'leq'

            constraint_id = str(uuid.uuid4())
            self.constraints[constraint_id]['var'] = decision_var
            self.constraints[constraint_id]['val'] = self.lower_lims[decision_var]
            self.constraints[constraint_id]['dir'] = 'geq'

    def split_region(self,approx_funcs,sig_funcs,opt_vars)->[]:

        split_dim = 0
        split_dim_err = 0

        for decision_var in range(self.dims):
            curr_err = sig_funcs[decision_var](opt_vars[decision_var])-approx_funcs[decision_var](opt_vars[decision_var])
            if curr_err > split_dim_err:
                split_dim_err = curr_err
                split_dim = decision_var

        new_upper_bounds_lower_subrectangle = copy.deepcopy(self.upper_lims)
        new_upper_bounds_lower_subrectangle[split_dim] = opt_vars[decision_var]

        new_lower_bounds_upper_subrectangle = copy.deepcopy(self.lower_lims)
        new_lower_bounds_upper_subrectangle[split_dim] = opt_vars[decision_var]

        return [search_region(self.lower_lims,new_upper_bounds_lower_subrectangle),search_region(new_lower_bounds_upper_subrectangle,self.upper_lims)]








def intersect_feasible_region_and_search_area(constraints,rectangle):
