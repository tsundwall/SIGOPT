import alg as so
import numpy as np
import scipy
# from examples import set_ubs
import examples as ex

def set_ubs(r,prev_alloc,c_arr):

    ubs = [r]*len(c_arr)
    c_sum = 0

    for i,x_i in enumerate(prev_alloc):
        if x_i > c_arr[i]:
            c_sum += c_arr[i]

    for i,x_i in enumerate(prev_alloc):
        if x_i > c_arr[i]:
            ubs[i] = r-c_sum+c_arr[i]


    return ubs

class bilevel_problem:

    def __init__(self,a,c,nu,eps,alpha,beta,inner_eps,inner_cross_eps,compute_lambda_shape=False,c_instance=0):

        self.inner_calls = 0
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
        self.time_start = time.time()
        self.time_lim = 100000
        self.compute_lambda_shape = compute_lambda_shape
        self.c_instance = c_instance


    def maximum_search(self,curr_r,k,r_max_prev,maxima_r):
        #print('FIND MAXIMUM')
        prev_r = None
        no_activation = True
        x_alloc_prev = None
        curr_lam = np.Inf
        prev_lam = -np.Inf
        self.sigopt_problem.r = curr_r
        self.sigopt_problem.init_upper_limits = [curr_r]*self.n
        self.sigopt_problem.clear()
        self.inner_calls += 1

        self.sigopt_problem.solve()
        grad_l = self.sigopt_problem.sol.dual_r
        l_value = self.sigopt_problem.sol.lb
        x_alloc = self.sigopt_problem.sol.x_opt_approx

        while no_activation and (time.time() - self.time_start < self.time_lim):
            # print(f'solving max,,,: {curr_r}')
            activation = self.activation_occurred(k,x_alloc) #this should never be true on the first iteration
            # #print(f'over it: {prev_r}')
            # #print(f'over it: {curr_r}')
            if activation:

                if self.activation_yielded_max(curr_r,curr_lam,prev_lam,grad_l):
                    print(f'max found: {prev_r}')

                    #print(x_alloc)
                    if x_alloc_prev is not None:
                        return x_alloc_prev,prev_r,prev_lam,False
                    else:#discouraged
                        print('CAREFUL')
                        return x_alloc,curr_r,prev_lam,False
                else:
                    print(f'max found: {prev_r}')
                    return x_alloc_prev,prev_r,prev_lam,True
            prev_r = curr_r
            # #print(f'GRAD: {self.alpha-self.beta*grad_l}')
            # self.nu = -(self.c[k-1]-(curr_r-maxima_r[-1]))/((self.a/4)-self.alpha)
            if k < self.n:
                curr_r = prev_r + np.min([.9*self.c[0]*np.abs(self.nu*(self.alpha-self.beta*grad_l)),self.c[k]])
            else:
                curr_r = prev_r + self.nu*(self.alpha-self.beta*grad_l)
            prev_lam = curr_lam
            curr_lam = self.eval_lambda(curr_r,l_value)

            self.all_sols.append(self.sigopt_problem.sol.lb)


                # #print('whte')

            if curr_r - r_max_prev < self.c[k-1]:
                # #print(x_alloc)
                x_alloc_prev = x_alloc
                grad_l, x_alloc, l_value = self.propogate_sol(k,x_alloc,curr_r,prev_r)

            else:
                self.sigopt_problem.r = curr_r
                if False:#curr_r > prev_r:
                    self.sigopt_problem.init_upper_limits = self.tighten_lims_r_increase(curr_r,x_alloc)
                    #print(self.sigopt_problem.init_upper_limits)
                else:
                    self.sigopt_problem.init_upper_limits = [curr_r]*self.n
                x_alloc_prev = x_alloc
                self.sigopt_problem.clear()
                self.inner_calls += 1
                self.sigopt_problem.solve()
                grad_l = self.sigopt_problem.sol.dual_r
                x_alloc = self.sigopt_problem.sol.x_opt_approx
                l_value = self.sigopt_problem.sol.lb
            # #print(x_alloc)
            # #print(f'dual: {self.sigopt_problem.sol.dual_r}')
            # #print(curr_r)
        if time.time() - self.time_start > self.time_lim:
            return None,None,None,None


    def get_dual_manually(self,k,r):
        offset = 0

        for i in range(k-1):
            offset += self.c[i]

        addtl_alloc = (r-offset)/(k-1)

        return self.sig_funcs[0].sigmoid_derivative(self.c[0]+addtl_alloc)

    def minimum_search(self,curr_r,k,r_max_prev):
        #print('FIND MINIMUM')
        highest_r_evaluated = self.c[k-1] + r_max_prev
        curr_lam = -np.Inf
        prev_lam = np.Inf
        prev_r = -np.Inf
        prev_r_2 = -np.Inf
        self.sigopt_problem.r = curr_r
        self.sigopt_problem.init_upper_limits = [curr_r]*self.n
        self.sigopt_problem.clear()
        self.inner_calls += 1
        self.sigopt_problem.solve()
        x_alloc = self.sigopt_problem.sol.x_opt_approx

        grad_l = self.sigopt_problem.sol.dual_r
        l_value = self.sigopt_problem.sol.lb

        if k > self.n:
            self.eps *= .01
            self.nu *= 2

        while True:
            # print(f'fog: {highest_r_evaluated}')
            # print(f'curr_r: {curr_r}')
            if np.abs(curr_lam-prev_lam) < self.eps:
                # print('shit')
                if True:#not ((curr_r > prev_r and curr_r > prev_r_2) or (curr_r < prev_r and curr_r < prev_r_2)):
                    break
            if time.time() - self.time_start > self.time_lim:
                break
            # #print(np.abs(curr_lam-prev_lam) > self.eps)
            prev_r_2 = prev_r
            prev_r = curr_r
            # #print(curr_r)
            curr_r = prev_r - self.nu*(self.alpha-self.beta*grad_l)
            prev_lam = curr_lam
            curr_lam = self.eval_lambda(curr_r,l_value)

            self.all_sols.append(self.sigopt_problem.sol.lb)

            if k > self.n or curr_r < highest_r_evaluated:
                x_alloc_prev = x_alloc
                # print('SKIP')
                grad_l, x_alloc, l_value = self.propogate_sol(k,x_alloc,curr_r,prev_r)
                # print(grad_l)

            else:
                self.sigopt_problem.r = curr_r
                if False:#curr_r > prev_r:
                    self.sigopt_problem.init_upper_limits = self.tighten_lims_r_increase(curr_r,x_alloc)
                else:
                    self.sigopt_problem.init_upper_limits = [curr_r]*self.n
                self.sigopt_problem.clear()
                self.inner_calls += 1
                self.sigopt_problem.solve()

                #grad_l = self.sigopt_problem.sol.dual_r
                x_alloc = self.sigopt_problem.sol.x_opt_approx
                if curr_r > highest_r_evaluated:
                    # print('SWITCH')
                    highest_r_evaluated = curr_r


                grad_l = self.get_dual_manually(k,curr_r)
                # print(grad_l)
                # print(x_alloc)
                l_value = self.sigopt_problem.sol.lb
            # print(self.sig_funcs[0].sigmoid_derivative(x_alloc[0]))
            # print(self.sig_funcs[1].sigmoid_derivative(x_alloc[1]))
            # except:
            #     #print('try')
            #     curr_r = curr_r + 0.1
            #     self.sigopt_problem.solve()

            # #print(f'dual: {self.sigopt_problem.sol.dual_r}')
            # #print(f'curr r: {curr_r}')
            # # #print(f'inner: {self.sigopt_problem.sol.ub}')
            # #print(f'outer: {self.eval_lambda(curr_r,self.sigopt_problem.sol.lb)}')
            # #print(f'gradient: {self.alpha-self.beta*self.sigopt_problem.sol.dual_r}')
            # # #print(f'conv: {curr_lam-prev_lam}')

            # #print(f'\n')
            # #print(curr_lam-prev_lam)
        print(f'min found: {curr_r}')
        # print(f'min found: {grad_l}')

        if time.time() - self.time_start > self.time_lim:

            return None,None,None
        return x_alloc,curr_r,curr_lam

    def tighten_lims_r_increase(self,r,x_alloc):

        return set_ubs(r,x_alloc,self.c)

    def is_first_minimum_above_second_activation():
        pass

    def perturb_local_max_sol(self,k,r_max_prev): #anything leq actual max
        # #print(self.c[k-1])
        if k <= self.n:
            return self.c[k-1]+r_max_prev
        return 1+r_max_prev

    def perturb_local_min_sol(self,r_max_prev,r_min_prev):
        # #print(r_max_prev)
        # #print(r_min_prev)
        # #print('OK')
        return r_max_prev-r_min_prev

    def activation_occurred(self,k,x_alloc):

        # #print('TEST')
        # #print(k)
        return x_alloc[k-1] > self.c[k-1]
        #return (x_alloc_prev[k-1] < self.c[k-1]) and (x_alloc[k-1] > self.c[k-1])

    def activation_yielded_max(self,curr_r,lam,lam_prev,grad_l): #only accurate on 'left' side of Lambda;

        return lam<lam_prev or self.alpha*curr_r - self.beta*grad_l > 0

    def propogate_sol(self,k,x_approx,curr_r,prev_r):


        l_value = self.eval_l(x_approx)
        # #print(f'fgef:{curr_r}')
        # #print(f'fgef:{prev_r}')
        x_approx_new = []
        for c_i,x_i in zip(self.c,x_approx):
            if x_i > c_i:
                x_i += (curr_r-prev_r)/(k-1)
            x_approx_new.append(x_i)

        grad_l = self.get_dual_manually(k,curr_r)
        # #print(x_approx_new)
        # #print(grad_l)
        return grad_l,x_approx_new,l_value

    def eval_l(self,x):
        # #print(x)
        l_value = 0
        for i,x_i in enumerate(x):
            l_value += self.sig_funcs[i].evaluate(x_i)

        return l_value

    def get_first_min(self):
        def deriv_first_var(x):
            return [(self.a*np.exp(x[0]+self.c[0]))/((np.exp(x[0])+np.exp(self.c[0]))**2) - (self.alpha/self.beta),x[0]-x[1]**2-self.c[0]]

        x_opt,_ = scipy.optimize.fsolve(deriv_first_var,[2*self.c[0],0.01])
        #print(f'first min: {x_opt}')
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

        path_str = f'{self.n}, {self.c_instance}, {self.alpha}, {self.inner_eps}, {self.time_lim}'

        if self.compute_lambda_shape:
            ex.test_sig_prog_simple(n=self.n,a=self.a,c=self.c,eps=self.inner_eps,lb=.1,ub=4*self.n,iters=10*self.n,path_str=path_str)

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

                #print(maxima_r)
                curr_r = self.perturb_local_max_sol(k=last_activation_idx,r_max_prev=maxima_r[-1])
                # #print('sec')

                curr_x_alloc, curr_r, curr_lam = self.minimum_search(curr_r=curr_r,k=last_activation_idx,r_max_prev=maxima[-1])

                if curr_lam is None:
                    #print('TIME EXPIRED')
                    argmin = np.argmin(minima)
                    return minima_x_allocs[argmin],minima_r[argmin],minima[argmin],minima,minima_r,maxima,maxima_r,self.inner_calls,time.time()-self.time_start


                # ##print(minima)
                # ##print(curr_lam)
                if curr_lam > np.min(minima):
                    argmin = np.argmin(minima)
                    return minima_x_allocs[argmin],minima_r[argmin],minima[argmin],minima,minima_r,maxima,maxima_r,self.inner_calls,time.time()-self.time_start

                all_active = True
                for x_i,c_i in zip(curr_x_alloc,self.c):
                    if x_i<c_i:
                        all_active= False
                # ##print(all_active)
                if all_active: #curr min must be global minimum if this condition holds
                    return curr_x_alloc, curr_r, curr_lam,minima,minima_r,maxima,maxima_r,self.inner_calls,time.time()-self.time_start

                else:
                    minima.append(curr_lam)
                    minima_r.append(curr_r)
                    minima_x_allocs.append(curr_x_alloc)

            if len(maxima) >= 1:

                #print(maxima_r)
                #print(minima_r)
                curr_r += self.perturb_local_min_sol(r_max_prev=maxima_r[-1],r_min_prev=minima_r[-2])

                curr_x_alloc, curr_r, curr_lam, is_min_min = self.maximum_search(curr_r=curr_r,k=last_activation_idx,r_max_prev=maxima[-1],maxima_r=maxima_r)

                #print(curr_r is None)
                if curr_lam is None:
                    #print('TIME EXPIRED')
                    argmin = np.argmin(minima)
                    return minima_x_allocs[argmin],minima_r[argmin],minima[argmin],minima,minima_r,maxima,maxima_r,self.inner_calls,time.time()-self.time_start

            else:
                curr_r = self.c[0]+self.c[1]
                #print(f'fr: {curr_r}')
                curr_x_alloc = [0]*self.n
                curr_x_alloc[0] = curr_r
                curr_lam = self.alpha*curr_r - self.beta*self.eval_l(curr_x_alloc)
                last_activation_idx = 2


            last_activation_idx += 1



            maxima.append(curr_lam)
            maxima_r.append(curr_r)
            maxima_x_allocs.append(curr_x_alloc)

            if is_min_min:
                return curr_x_alloc, curr_r, curr_lam,minima,minima_r,maxima,maxima_r,self.inner_calls,time.time()-self.time_start
            # prev_r = np.Inf
            # prev_lam = np.Inf
            # curr_r += .75
            # minima.append(self.eval_lambda(curr_r))

            # if len(minima) > 1 and minima[-1] > minima[-1]:
            #     #print('DONE')
            #     global_min_found = True

            # while np.abs(curr_lam-prev_lam) > (self.eps*.03):
            #     #print(curr_r)
            #     self.sigopt_problem.r = curr_r
            #     self.sigopt_problem.clear()

            #     try:
            #         self.sigopt_problem.solve()
            #         self.all_sols.append(self.sigopt_problem.sol.ub)
            #     except:
            #         #print('try')
            #     #     #print(curr_r)
            #         self.sigopt_problem.clear()
            #     #     curr_r = curr_r - 0.005
            #         self.sigopt_problem.r -= 0.1
            #     #     #print(curr_r)
            #         self.sigopt_problem.solve()
            #         self.all_sols.append(self.sigopt_problem.sol.ub)

            #     # #print('\n')

                # #print('process')
                # prev_r = curr_r
                # curr_r = prev_r + self.nu*(self.alpha-self.beta*self.sigopt_problem.sol.parent_dual_r)
                # prev_lam = curr_lam
                # curr_lam = self.eval_lambda(curr_r)
                # # #print(f'r{self.sigopt_problem.sol.parent_dual_r}')
                # #print(f'curr r: {curr_r}')
                # # #print(f'inner: {self.sigopt_problem.sol.ub}')
                # #print(f'outer: {self.eval_lambda(curr_r)}')
                # #print(f'gradient: {self.alpha-self.beta*self.sigopt_problem.sol.parent_dual_r}')
                # #print(f'Convergernce: {np.abs(curr_lam-prev_lam)}')
                # #print(f'prev lam: {prev_lam}')

            #     #print(f'\n')
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
# c = [1.68755356, 1.69657806, 1.16548715, 2.15513042, 1.91748372,
#        2.97135358, 1.90930468, 2.1212319 , 2.28685834, 2.30011484,
#        1.30609675, 1.24225479, 2.0825002 , 2.64996499, 1.14816546,
#        1.14374299, 2.95539075, 1.76307915, 2.97550369, 2.40989722,
#        2.59114533, 1.76773409, 2.00224763, 2.70317089, 1.20777725,
#        2.24525831, 2.89310534, 2.75941663, 2.39219145, 2.35303925]
# sig_funcs = build_sig_funcs_simple(a,c)
np.random.seed(158)
c = np.random.uniform(low=1, high=3, size=(30,))
import time

# pr = so.problem(sig_funcs,[],[0]*10,upper_lims,0.1,1,d_vec=[1,1,1,1,1,1,1,1,1,1])
t = time.time()
blvl = bilevel_problem(a,c,4,0.00001,.22,1,.0001,.01,False,1)

opt_x_alloc, opt_r, opt_lam,minima,minima_r,maxima,maxima_r,inner_calls,time_elapsed = blvl.solve()
print(time.time()-t)
print(opt_r)
print(minima)
print(minima_r)
print(maxima)
print(maxima_r)
print(inner_calls)
print(time_elapsed)
#print(blvl.all_sols)