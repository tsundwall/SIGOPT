# !pip install cvxpy
import cvxpy as cp
import numpy as np
from queue import PriorityQueue
import copy
import uuid
import time

class problem:

    def __init__(self,sig_funcs,constraints,init_upper_limits,tol,r,d_vec=None,A=None,b=None):
        self.sig_funcs = sig_funcs
        self.constraints = constraints
        self.init_upper_limits = init_upper_limits
        self.sol = None
        self.tol = tol
        self.max_approx_err = np.Inf
        self.r = r
        self.n = len(self.sig_funcs)
        self.d_vec = d_vec
        self.A = A
        self.b = b
        self.iters = 0

        if self.d_vec is None:
            self.d_vec = [1]*self.n

    def clear(self):
        self.sol = None
        self.max_approx_err = np.Inf

    def solve(self):

        t = time.time()

        q = bb_queue()
        q.add([search_region(self,self.n,np.zeros(self.n),self.init_upper_limits,[])])

        while q.length > 0 and self.max_approx_err > self.tol:

            curr_candidate = q.pop()

            self.max_approx_err = curr_candidate.evaluate_region()

            new_regions = curr_candidate.split_region()
            q.prune_split_regions(new_regions[0],new_regions[1])

        self.sol = q.pop()
        # if q.length == 0:
            # print('CRY')
        print(f'complete: {time.time()-t} secs')



class sigmoidal_function:

    def __init__(self,mu,lam,theta):
        self.a = mu
        self.b = lam
        self.c = theta

    def evaluate(self,val):

        return self.a / (1 + np.exp(-self.b*val +self.c))

    def sigmoid_derivative(self,x):
        return (self.a*self.b*(np.exp(self.b*x+self.c)))/(np.exp(self.b*x)+np.exp(self.c))**2


class piecewise_linear_bound:

    def __init__(self,sigmoid,l,u):
        self.lower_constraint = l
        self.upper_constraint = u
        self.sigmoid = sigmoid
        self.plb_arr = []

    def plb_get_tangent_line(self,sig,x):
        dx = sig.sigmoid_derivative(x)
        fx = sig.evaluate(x)

        return [dx, (-x*dx+fx)]

    def get_bisection(self,epsilon,mu,tol): #epsilon dictates the convergence; using 10^3 as tolerance
        tcvx = time.time()
        candidate_w = self.upper_constraint
        not_found=True

        while not_found:

            w_slope = self.sigmoid.sigmoid_derivative(candidate_w)

            f_w = self.sigmoid.evaluate(candidate_w)
            f_l = self.sigmoid.evaluate(self.lower_constraint)
            l_slope = (f_w - f_l) / (candidate_w-self.lower_constraint)

            if np.round(l_slope,tol) > np.round(w_slope,tol): #too far
                candidate_w = candidate_w *(1-epsilon)

            elif np.round(l_slope,tol) < np.round(w_slope,tol): #not far enough
                candidate_w = candidate_w *(1+epsilon)

            else:
                not_found = False

            epsilon = mu*epsilon

        w_intercept = f_l - (f_w - f_l)/(candidate_w-self.lower_constraint)*self.lower_constraint
        # print(time.time()-tcvx)
        return candidate_w,w_slope,w_intercept

    def plb_find_intersect(self,eq1,eq2): #intersection of two linearly independent lines

        s1,int1 = eq1
        s2,int2 = eq2

        return (int2-int1) / (s1-s2)

    def plb_get_minimum_val(self,lines,x):

        curr_min_val = lines[0][0]*x + lines[0][1]

        for line in lines[1:]:

            if line[0]*x + line[1] < curr_min_val:
                curr_min_val = line[0]*x + line[1]

        return curr_min_val


    def find_plb(self,tol):
        lines = []
        errors = []
        w,w_slope,w_intercept = self.get_bisection(.01,1,4)

        if w >= self.upper_constraint:

            f_l = self.sigmoid.evaluate(self.lower_constraint)
            f_u = self.sigmoid.evaluate(self.upper_constraint)

            slope = (f_u-f_l)/(self.upper_constraint-self.lower_constraint)
            self.plb_arr = [[slope,-slope*self.lower_constraint+f_l]]

            return

        lines.append([w_slope,w_intercept])

        lines.append(self.plb_get_tangent_line(self.sigmoid,self.upper_constraint))

        over_tolerance = True

        while over_tolerance:
            curr_max = 0
            curr_max_intersect = 0
            for i in range(len(lines)):
                for j in range(i,len(lines)):
                    if i != j:
                        intersect = self.plb_find_intersect(lines[i],lines[j])
                        val = self.plb_get_minimum_val(lines, intersect) - self.sigmoid.evaluate(intersect)

                        if val > curr_max:
                            curr_max = val
                            curr_max_intersect = intersect

            errors.append(curr_max)

            if curr_max < tol:
                over_tolerance = False

            else:
                lines.append(self.plb_get_tangent_line(self.sigmoid,curr_max_intersect))

        self.plb_arr = lines

class search_region:

    def __init__(self,problem,dims:int,lower_lims:[],upper_lims:[],x_opt_approx,lb=-np.Inf,ub=-np.Inf,parent_lims=[],parent_id=None,parent_dual_r=np.Inf,parent_split_dim=None):
        self.problem = problem
        self.dims = dims
        self.upper_lims = upper_lims
        self.lower_lims = lower_lims
        self.ub = ub
        self.lb = lb
        self.sig_funcs = self.problem.sig_funcs
        self.plb_funcs = []
        self.convex_flags = [True*self.dims]
        self.x_opt_approx = x_opt_approx
        self.x_type = 'prev'
        self.parent_lims = parent_lims
        self.id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.dual_r = np.Inf
        self.parent_dual_r = parent_dual_r
        self.parent_split_dim = parent_split_dim
        # self.constraints = {}

    def __lt__(self, other):
        return self.ub < other.ub

    def peek(self):

        # print(f'x alloc: {self.x_opt_approx}')
        print(f'ub - lb: {self.ub - self.lb}')
        # print(f'upper bound: {self.ub}')
        # print(f'lower bound: {self.lb}')
        # print(f'lower_lims: {self.lower_lims}')
        # print(f'upper_lims: {self.upper_lims}')
        # print(f'parent split dim: {self.parent_split_dim}')
        # print(f'parent lims: {self.parent_lims}')
        # print(f'id: {self.id}')
        # print(f'parent id: {self.parent_id}')
        print('\n')

    def evaluate_region(self):

        for func_idx,func in enumerate(self.sig_funcs):

            plb_obj = piecewise_linear_bound(func,self.lower_lims[func_idx],self.upper_lims[func_idx])
            plb_obj.find_plb(0.01)
            self.plb_funcs.append(plb_obj)

        self.solve_plb_approximation(self.problem.r)

        return np.abs(self.ub-self.lb)


    def eval_plb(self,idx):
        curr_plb = self.plb_funcs[idx]
        return curr_plb.plb_get_minimum_val(curr_plb.plb_arr,self.x_opt_approx[idx])

    def solve_plb_approximation(self,r):

        self.problem.iters += 1
        constraints = []

        n = len(self.plb_funcs)
        x = cp.Variable(n)
        t = cp.Variable(n)
        ones_vec = np.ones(n)

        for func_idx,function in enumerate(self.plb_funcs):

            pwl_arr = function.plb_arr

            constraints += [x[func_idx] >= self.lower_lims[func_idx]]
            constraints += [x[func_idx] <= self.upper_lims[func_idx]]

            if len(pwl_arr) != 0:

                for pwl_idx,pwl_line in enumerate(pwl_arr):
                    constraints += [pwl_line[0] * x[func_idx] + pwl_line[1] >= t[func_idx]]

        constraints += [self.problem.d_vec@x <= r]
        prob = cp.Problem(cp.Minimize(ones_vec@-t),
                         constraints)

        prob.solve()

        self.dual_r = constraints[-1].dual_value
        self.ub = -prob.value
        self.x_opt_approx = x.value
        self.lb = self.evaluate_sigmoid_lb(r)
        self.type= 'curr'

    def evaluate_sigmoid_lb(self,r):

        # tot_allocated = 0
        # lb = 0
        # idx = 0
        # end = False

        # while not end and idx<self.dims:

        #     next_upper_constraints =  self.upper_lims[0]

        #     tot_allocated += next_upper_constraints
        #     if tot_allocated >= r:
        #         next_upper_constraints -= tot_allocated - r
        #         end = True

        #     lb += self.plb_funcs[idx].sigmoid.evaluate(next_upper_constraints)
        #     idx += 1

        lb = 0

        for func_idx,sig in enumerate(self.sig_funcs):
            lb += sig.evaluate(self.x_opt_approx[func_idx])

        return lb

    # def construct_constraints(self):

    #     self.constraints = {}

    #     for decision_var in range(self.dims):
    #         constraint_id = str(uuid.uuid4())
    #         self.constraints[constraint_id]['var'] = decision_var
    #         self.constraints[constraint_id]['val'] = self.upper_lims[decision_var]
    #         self.constraints[constraint_id]['dir'] = 'leq'

    #         constraint_id = str(uuid.uuid4())
    #         self.constraints[constraint_id]['var'] = decision_var
    #         self.constraints[constraint_id]['val'] = self.lower_lims[decision_var]
    #         self.constraints[constraint_id]['dir'] = 'geq'

    def split_region(self)->[]:

        split_dim = 0
        split_dim_err = 0

        for decision_var in range(self.dims):
            curr_err = np.abs(self.sig_funcs[decision_var].evaluate(self.x_opt_approx[decision_var])-self.eval_plb(decision_var))

            if curr_err > split_dim_err:
                split_dim_err = curr_err
                split_dim = decision_var

        new_upper_bounds_lower_subrectangle = copy.deepcopy(self.upper_lims)

        new_upper_bounds_lower_subrectangle[split_dim] = self.x_opt_approx[split_dim]
        # print(f'splitdim: {self.x_opt_approx}')
        # print(f'q: {new_upper_bounds_lower_subrectangle[split_dim]}')
        new_lower_bounds_upper_subrectangle = copy.deepcopy(self.lower_lims)
        new_lower_bounds_upper_subrectangle[split_dim] = self.x_opt_approx[split_dim]

        return [search_region(self.problem,self.dims,self.lower_lims,new_upper_bounds_lower_subrectangle,self.x_opt_approx,self.lb,self.ub,[self.lower_lims,self.upper_lims],self.id,self.dual_r,split_dim),search_region(self.problem,self.dims,new_lower_bounds_upper_subrectangle,self.upper_lims,self.x_opt_approx,self.lb,self.ub,[self.lower_lims,self.upper_lims],self.id,self.dual_r,split_dim)]

class bb_queue:

    def __init__(self) -> None:
        self.queue = PriorityQueue()
        self.max_lower_bound = -np.Inf
        self.length = 0

    def add(self,search_regions):

        for region in search_regions:
            # region.peek()

            self.queue.put((-region.ub,region))


        self.length += 1
        # print(self.queue.queue)

    def prune_split_regions(self,lower_region,upper_region):

        if lower_region.ub >= self.max_lower_bound:

            self.add([lower_region])
            if lower_region.lb > self.max_lower_bound:
                self.max_lower_bound = lower_region.lb
            self.length += 1

        if upper_region.ub >= self.max_lower_bound:

            self.add([upper_region])
            if upper_region.lb > self.max_lower_bound:
                self.max_lower_bound = upper_region.lb
            self.length += 1

    def pop(self):
        candidate_region_not_found = True

        while candidate_region_not_found:

            if len(self.queue.queue) == 0:
                return "No candidates remain"

            priority,candidate = self.queue.get()
            # print(f'line:{priority}')

            self.length -= 1
            if candidate.ub >= self.max_lower_bound:

                candidate_region_not_found = False

        return candidate

