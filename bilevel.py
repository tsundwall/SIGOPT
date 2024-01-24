import alg as so
import numpy as np

class bilevel_problem: #will be quasiconvex

    def __init__(self,sigopt_problem,nu,eps,alpha,beta):
        self.sigopt_problem = sigopt_problem
        self.all_sols = []
        self.r_range = [1,13]
        self.nu = nu
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def solve(self):

        curr_r = self.r_range[1]
        prev_r = np.Inf

        while np.abs(curr_r-prev_r) > self.eps:

            self.sigopt_problem.r = curr_r
            self.sigopt_problem.clear()
            print('ok')
            self.sigopt_problem.solve()
            # print('\n')
            self.all_sols.append(self.sigopt_problem.sol.ub)
            print('process')
            prev_r = curr_r
            curr_r = prev_r - self.nu*(self.alpha+self.beta*self.sigopt_problem.sol.parent_dual_r)
            # print(f'r{self.sigopt_problem.sol.parent_dual_r}')
            print(f'curr r: {curr_r}')
            print(f'inner: {self.sigopt_problem.sol.ub}')
            print(f'outer: {self.eval_lambda(curr_r)}')
            print(f'lambda: {self.sigopt_problem.sol.parent_dual_r}')
            print(np.abs(curr_r-prev_r))
            print(f'\n')
        self.opt = curr_r


    def eval_lambda(self,r):
        return self.alpha*r + self.beta*(-self.sigopt_problem.sol.ub)

def build_sig_funcs(a_arr,b_arr,c_arr):
    n = len(a_arr)
    funcs = []

    for i in range(n):
        funcs.append(so.sigmoidal_function(a_arr[i],b_arr[i],c_arr[i]))

    return funcs

a_arr = [1,1,1]#,1,1,1,1,1,1,1]
c_arr = [3,7,4]#,1,1,1,2,2,1,1]
b_arr = [1,1,1]#,1,1,1,1,1,1,1]#[4,2,1,4,6,4,1,1,2,1]

sig_funcs = build_sig_funcs(a_arr,b_arr,c_arr)

upper_lims = [10]*3

pr = so.problem(sig_funcs,[],upper_lims,0.05,20)
blvl = bilevel_problem(pr,.9,.1,.1,.6)
blvl.solve()
print(blvl.all_sols)
