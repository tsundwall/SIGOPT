import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import alg as so
import numpy as np

#add constraints
#testing


def build_sig_funcs(a_arr,b_arr,c_arr):
    n = len(a_arr)
    funcs = []

    for i in range(n):
        funcs.append(so.sigmoidal_function(a_arr[i],b_arr[i],c_arr[i]))

    return funcs

a_arr = [1,1,1,2,1,1,1,1,1,1,1,1,2,1,1,1,9,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1]
c_arr = [2,4,3,4.1,6.7,1,2,3,5,3,2,5,3,2,1,1,5,5,8,3,2,1,1,1,2,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1]#,2,1,2,3,4,5,6,7,8,7.5,6.5,4,3,3,2,1]
b_arr = [1,1,1,1,1,1,1,1,3,1,1,1,3,3,2,2,4,6,3,1,1,1,1,1,2,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1]

sig_funcs = build_sig_funcs(a_arr,b_arr,c_arr)

upper_lims = [10]*42
lower_lims = [0]*42
print(len(b_arr))
pr = so.problem(sig_funcs,[],lower_lims,upper_lims,.01,8,[2,1,2,1,2,2,4,0.6,1 ,1 ,4,6 ,1 ,1 ,2.3 ,3.3 ,4.1 ,2.6 ,1 ,1 ,1,1,1,1,1,2,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,1])

pr.solve()

print(np.round(pr.sol.x_opt_approx,4))
print(f'ub: {pr.sol.ub}')
print(f'lb: {pr.sol.lb}')
# print(pr.sol.parent_dual_r)
# print(pr.sol.x_type)
print(f'iters: {pr.iters}')
# print(pr.tot_cvx_time)
# print(pr.max_approx_err)
# print(pr.q)
# pr.get_theoretical_bounds()
# print(pr.bounds)

def eval(sigs,vals):
    sol = 0
    for i,sig in enumerate(sigs):
        sol += sig.evaluate(vals[i])
    return sol
