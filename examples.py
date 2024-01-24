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

a_arr = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
c_arr = [3.1,3.2,3,4.1,6.7,2,1,2,3,4,5,6,7,8,7.5,6.5,4,3,3,2,1]
b_arr = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

sig_funcs = build_sig_funcs(a_arr,b_arr,c_arr)

upper_lims = [10]*21

pr = so.problem(sig_funcs,[],upper_lims,0.001,20,[2,1,3,1,4,1,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1])

pr.solve()

print(pr.sol.x_opt_approx)
print(f'ub: {pr.sol.ub}')
print(f'lb: {pr.sol.lb}')
print(pr.sol.parent_dual_r)
print(pr.sol.x_type)
print(pr.iters)