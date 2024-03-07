import alg as so
import numpy as np
import scipy

class bilevel_problem:

    def __init__(self,a,c,nu,eps,alpha,beta,inner_eps,inner_cross_eps):

        self.all_sols = []
        self.n = len(c)
        self.inner_eps = inner_eps
        self.inner_cross_eps = inner_cross_eps
        # self.r_range = [50,13]
        self.nu = nu #learning rate
        self.eps = eps #for gradient descent
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.a = a
        self.sig_funcs = build_sig_funcs_simple(self.a,self.c)
        self.sigopt_problem = so.problem(self.sig_funcs,[],[0]*self.n,[0]*self.n,self.inner_eps,self.inner_cross_eps,0,d_vec=[1]*self.n)


    def maximum_search(self,curr_r,k,r_max_prev):
        print('FIND MAXIMUM')
        no_activation = True
        x_alloc_prev = None
        curr_lam = np.Inf
        prev_lam = -np.Inf
        self.sigopt_problem.r = curr_r
        self.sigopt_problem.init_upper_limits = [curr_r]*self.n
        self.sigopt_problem.clear()
        self.sigopt_problem.solve()
        grad_l = self.sigopt_problem.sol.dual_r
        l_value = self.sigopt_problem.sol.lb
        x_alloc = self.sigopt_problem.sol.x_opt_approx

        while no_activation:

            prev_r = curr_r
            if k < self.n:
                curr_r = prev_r + np.min([self.nu*(self.alpha-self.beta*grad_l),self.c[k]])
            else:
                curr_r = prev_r + self.nu*(self.alpha-self.beta*grad_l)
            prev_lam = curr_lam
            curr_lam = self.eval_lambda(curr_r,l_value)

            self.all_sols.append(self.sigopt_problem.sol.lb)

            if x_alloc_prev is not None:
                print('whte')
                activation = self.activation_occurred(k,x_alloc,x_alloc_prev)
                if activation:

                    if self.activation_yielded_max(curr_lam,prev_lam):
                        return x_alloc_prev,prev_r,curr_lam,False
                    else:
                        return x_alloc_prev,prev_r,curr_lam,True

            if curr_r - r_max_prev < self.c[k-1]:
                x_alloc_prev = x_alloc
                grad_l, x_alloc, l_value = self.propogate_sol(self.last_activation_idx,x_alloc,curr_r,prev_r)

            else:
                self.sigopt_problem.r = curr_r
                self.sigopt_problem.init_upper_limits = [curr_r]*self.n
                x_alloc_prev = x_alloc
                self.sigopt_problem.clear()
                self.sigopt_problem.solve()
                grad_l = self.sigopt_problem.sol.dual_r
                x_alloc = self.sigopt_problem.sol.x_opt_approx
                l_value = self.sigopt_problem.sol.lb
            print(x_alloc)
            print(f'dual: {self.sigopt_problem.sol.dual_r}')





    def minimum_search(self,curr_r,k,r_max_prev):
        print('FIND MINIMUM')
        curr_lam = -np.Inf
        prev_lam = np.Inf
        self.sigopt_problem.r = curr_r
        self.sigopt_problem.init_upper_limits = [curr_r]*self.n
        self.sigopt_problem.clear()
        self.sigopt_problem.solve()
        x_alloc = self.sigopt_problem.sol.x_opt_approx
        grad_l = self.sigopt_problem.sol.dual_r
        l_value = self.sigopt_problem.sol.lb

        if k > self.n:
            self.eps *= .01
            self.nu *= 2

        while np.abs(curr_lam-prev_lam) > self.eps:

            print(np.abs(curr_lam-prev_lam) > self.eps)
            prev_r = curr_r
            curr_r = prev_r - self.nu*(self.alpha-self.beta*grad_l)
            prev_lam = curr_lam
            curr_lam = self.eval_lambda(curr_r,l_value)

            self.all_sols.append(self.sigopt_problem.sol.lb)

            if k > self.n or curr_r - r_max_prev < self.c[k-1]:
                x_alloc_prev = x_alloc
                grad_l, x_alloc, l_value = self.propogate_sol(k,x_alloc,curr_r,prev_r)

            else:
                self.sigopt_problem.r = curr_r
                self.sigopt_problem.init_upper_limits = [curr_r]*self.n
                self.sigopt_problem.clear()
                self.sigopt_problem.solve()
                grad_l = self.sigopt_problem.sol.dual_r
                x_alloc = self.sigopt_problem.sol.x_opt_approx
                l_value = self.sigopt_problem.sol.lb

            # except:
            #     print('try')
            #     curr_r = curr_r + 0.1
            #     self.sigopt_problem.solve()

            print(f'dual: {self.sigopt_problem.sol.dual_r}')
            print(f'curr r: {curr_r}')
            # print(f'inner: {self.sigopt_problem.sol.ub}')
            print(f'outer: {self.eval_lambda(curr_r,self.sigopt_problem.sol.lb)}')
            print(f'gradient: {self.alpha-self.beta*self.sigopt_problem.sol.dual_r}')
            # print(f'conv: {curr_lam-prev_lam}')

            print(f'\n')
            print(curr_lam-prev_lam)

        return x_alloc,curr_r,curr_lam


    def perturb_local_max_sol(self,k,r_max_prev): #anything leq actual max
        # print(self.c[k-1])
        if k <= self.n:
            return self.c[k-1]+r_max_prev
        return 1+r_max_prev

    def perturb_local_min_sol(self,r_max_prev,r_min_prev):
        print(r_max_prev)
        print(r_min_prev)
        print('OK')
        return r_max_prev-r_min_prev

    def activation_occurred(self,k,x_alloc,x_alloc_prev):
        print(x_alloc_prev)
        print(x_alloc)
        print(k)
        return (x_alloc_prev[k-1] < self.c[k-1]) and (x_alloc[k-1] > self.c[k-1])

    def activation_yielded_max(self,lam,lam_prev): #only accurate on 'left' side of Lambda;
        return lam<lam_prev

    def propogate_sol(self,k,x_approx,curr_r,prev_r):


        l_value = self.eval_l(x_approx)
        print(f'fgef:{curr_r}')
        print(f'fgef:{prev_r}')
        x_approx_new = []
        for x_i in x_approx:
            x_i += (curr_r-prev_r)/(k-1)
            x_approx_new.append(x_i)

        grad_l = self.sig_funcs[0].sigmoid_derivative(x_approx[0])
        print(x_approx_new)
        return grad_l,x_approx_new,l_value

    def eval_l(self,x):
        print(x)
        l_value = 0
        for i,x_i in enumerate(x):
            l_value += self.sig_funcs[i].evaluate(x_i)

        return l_value

    def get_first_min(self):
        def deriv_first_var(x):
            return [(self.a*np.exp(x[0]+self.c[0]))/((np.exp(x[0])+np.exp(self.c[0]))**2) - (self.alpha/self.beta),x[0]-x[1]**2-self.c[0]]

        x_opt,_ = scipy.optimize.fsolve(deriv_first_var,[2*self.c[0],0.01])

        x_alloc = [0]*self.n
        x_alloc[0] = x_opt

        return x_alloc,self.alpha*x_opt-self.beta*self.eval_l(x_alloc),x_opt

    # def general_search(self):
    #     minima = []
    #     minima_r = []
    #     minima_x = []
    #     max_minima = (self.sigopt_problem.n*(self.sigopt_problem.n+1))/2
    #     curr_r = self.r_range[0]
    #     curr_lam = -np.Inf
    #     prev_lam = np.Inf
    #     last_minimum = False

    #     while minima < max_minima:

    #         while np.abs(curr_lam-prev_lam) > self.eps:

    #             self.sigopt_problem.r = curr_r
    #             self.sigopt_problem.clear()
    #             self.sigopt_problem.solve()
    #             self.all_sols.append(self.sigopt_problem.sol.ub)

    #             prev_r = curr_r
    #             curr_r = prev_r - self.nu*(self.alpha-self.beta*self.sigopt_problem.sol.parent_dual_r)

    #             prev_lam = curr_lam
    #             curr_lam = self.eval_lambda(curr_r)

    #         minima.append(self.eval_lambda(curr_r))
    #         minima_r.append(curr_r)
    #         minima_x.append(self.sigopt_problem.sol.x_opt_approx)

    #         if last_minimum:
    #             argmin = np.argmin(minima)
    #             return {'obj value':np.min(minima),'optimal (outer)':minima_r,'optimal (inner)':minima_x}


    #         prev_r = np.Inf
    #         prev_lam = np.Inf
    #         curr_r += .75

    #         while np.abs(curr_lam-prev_lam) > (self.eps):

    #             self.sigopt_problem.r = curr_r
    #             self.sigopt_problem.clear()

    #             self.sigopt_problem.solve()
    #             self.all_sols.append(self.sigopt_problem.sol.ub)

    #             prev_r = curr_r
    #             curr_r = prev_r + self.nu*(self.alpha-self.beta*self.sigopt_problem.sol.parent_dual_r)
    #             prev_lam = curr_lam
    #             curr_lam = self.eval_lambda(curr_r)

    #             curr_x = self.sigopt_problem.sol.x_opt_approx

    #             if np.all(np.greater(curr_x,self.sigopt_problem.c_arr)):
    #                 if self.sigopt_problem.sol.parent_dual_r < (self.alpha/self.beta):
    #                     argmin = np.argmin(minima)
    #                     return {'obj value':np.min(minima),'optimal (outer)':minima_r[argmin],'optimal (inner)':minima_x[argmin]}
    #                 else:
    #                     last_minimum = True




    def solve(self):
        minima = []
        minima_r = []
        minima_x_allocs = []
        maxima = []
        maxima_r = []
        maxima_x_allocs = []
        self.c.sort()
        curr_r = 0
        last_activation_idx = 1
        prev_r = np.Inf
        global_min_found = False
        is_min_min = False
        while not global_min_found:

            if curr_r == 0:
                curr_x_alloc,curr_lam,curr_r = self.get_first_min()
                minima.append(curr_lam)
                minima_r.append(curr_r)
                minima_x_allocs.append(curr_x_alloc)

            else:
                curr_r = self.perturb_local_max_sol(k=last_activation_idx,r_max_prev=maxima_r[-1])
                print('sec')
                curr_x_alloc, curr_r, curr_lam = self.minimum_search(curr_r=curr_r,k=last_activation_idx,r_max_prev=maxima[-1])
                print('sizzupr')
                print(minima)
                print(curr_lam)
                if curr_lam > np.min(minima):
                    argmin = np.argmin(minima)
                    return minima_x_allocs[argmin],minima[argmin],minima_r[argmin]

                all_active = True
                for x_i,c_i in zip(curr_x_alloc,self.c):
                    if x_i<c_i:
                        all_active= False
                print(all_active)
                if all_active: #curr min must be global minimum if this condition holds
                    return curr_x_alloc, curr_r, curr_lam

                else:
                    minima.append(curr_lam)
                    minima_r.append(curr_r)
                    minima_x_allocs.append(curr_x_alloc)

            if len(maxima) >= 1:
                curr_r += self.perturb_local_min_sol(r_max_prev=maxima_r[-1],r_min_prev=minima_r[-2])
                curr_x_alloc, curr_r, curr_lam, is_min_min = self.maximum_search(curr_r=curr_r,k=last_activation_idx,r_max_prev=maxima[-1])

            else:
                curr_r = self.c[0]+self.c[1]
                curr_x_alloc = [0]*self.n
                curr_x_alloc[0] = curr_r
                curr_lam = self.alpha*curr_r - self.beta*self.eval_l(curr_x_alloc)
                last_activation_idx = 2


            last_activation_idx += 1



            maxima.append(curr_lam)
            maxima_r.append(curr_r)
            maxima_x_allocs.append(curr_x_alloc)

            if is_min_min:
                return curr_x_alloc, curr_r, curr_lam
            # prev_r = np.Inf
            # prev_lam = np.Inf
            # curr_r += .75
            # minima.append(self.eval_lambda(curr_r))

            # if len(minima) > 1 and minima[-1] > minima[-1]:
            #     print('DONE')
            #     global_min_found = True

            # while np.abs(curr_lam-prev_lam) > (self.eps*.03):
            #     print(curr_r)
            #     self.sigopt_problem.r = curr_r
            #     self.sigopt_problem.clear()

            #     try:
            #         self.sigopt_problem.solve()
            #         self.all_sols.append(self.sigopt_problem.sol.ub)
            #     except:
            #         print('try')
            #     #     print(curr_r)
            #         self.sigopt_problem.clear()
            #     #     curr_r = curr_r - 0.005
            #         self.sigopt_problem.r -= 0.1
            #     #     print(curr_r)
            #         self.sigopt_problem.solve()
            #         self.all_sols.append(self.sigopt_problem.sol.ub)

            #     # print('\n')

                # print('process')
                # prev_r = curr_r
                # curr_r = prev_r + self.nu*(self.alpha-self.beta*self.sigopt_problem.sol.parent_dual_r)
                # prev_lam = curr_lam
                # curr_lam = self.eval_lambda(curr_r)
                # # print(f'r{self.sigopt_problem.sol.parent_dual_r}')
                # print(f'curr r: {curr_r}')
                # # print(f'inner: {self.sigopt_problem.sol.ub}')
                # print(f'outer: {self.eval_lambda(curr_r)}')
                # print(f'gradient: {self.alpha-self.beta*self.sigopt_problem.sol.parent_dual_r}')
                # print(f'Convergernce: {np.abs(curr_lam-prev_lam)}')
                # print(f'prev lam: {prev_lam}')

            #     print(f'\n')
            # self.opt = curr_r
            # min_minima_not_found = False


    def eval_lambda(self,r,l_value):
        return self.alpha*r - self.beta*l_value

def build_sig_funcs_simple(a,c_arr):
    n = len(c_arr)
    funcs = []
    c_arr.sort()
    for i in range(n):
        funcs.append(so.sigmoidal_function(a,1,c_arr[i]))

    return funcs

a = 1
c = [1,2,3,4]
# sig_funcs = build_sig_funcs_simple(a,c)

import time

# pr = so.problem(sig_funcs,[],[0]*10,upper_lims,0.1,1,d_vec=[1,1,1,1,1,1,1,1,1,1])
t = time.time()
blvl = bilevel_problem(a,c,5,0.001,.13,1,.001,.005)
blvl.solve()
print(time.time()-t)
# print(blvl.all_sols)