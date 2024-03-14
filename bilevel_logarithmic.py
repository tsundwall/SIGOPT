import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

class problem:
    def __init__(self,d:list[float],a:list[float],b:list[float],c:list[float],l:list[float],alpha:float,beta:float,eps:float,tau:float) -> None:
        self.d = d
        self.a = a
        self.b = b
        self.c = c
        self.l = l
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tau = tau
        self.n = len(self.d)

    def deriv(self,x,i):
        return (self.a[i]*self.b[i])/(self.b[i]*x+self.c[i])

    def log_inverse(self,x):
        ttl = 0
        all_feasible = True
        x_vals = []
        for i in range(self.n):

         val = (self.a[i]*self.b[i]-self.c[i]*self.d[i]*x)/(self.b[i]*self.d[i]*x)
         x_vals.append(val)
         ttl += val*self.d[i]
        #  print(val)
         if val < self.l[i]:
            #  print('damn')
             all_feasible = False

        return ttl,x_vals,all_feasible

    def check_mult_at_l(self):
        all = []

        for i,l_i in enumerate(self.l):
            all.append(self.deriv(l_i,i)/self.d[i])

        return np.max(all),np.argmax(all)

    def lam(self,r_t):
        sol,multipliers,x_vars = self.L_hat(r_t)
        return self.alpha*r_t + self.beta*(-sol),sol,multipliers,x_vars


    def L_hat(self,r):
        # print('call')
        # n = 5
        # A = -np.array([[1,.2,.2,.1,1]])
        # b = -np.array([m])
        # c = np.array([1.1,1.2,2.3,1.4,9.5])

        x = cp.Variable(self.n)
        # print(-self.d)
        # print(-r)
        constraints = [-self.d @ x == -r]

        for i,l_i in enumerate(self.l):

            constraints += [x[i] >= l_i]

        prob = cp.Problem(cp.Minimize(-(self.a.T@cp.log(cp.multiply(self.b.T,x)+self.c))),
                        constraints)

        sol = prob.solve()
        # print(r)
        multipliers = constraints[0].dual_value
        # print(multipliers)
        return -sol,multipliers,x.value

    def solve(self):
        start_grad, start_grad_idx = self.check_mult_at_l()
        if start_grad < self.alpha/self.beta:
            return 0,[0]*self.n,0
        # print(self.c[start_grad_idx])
        # print(self.a[start_grad_idx])
        # print(self.b[start_grad_idx])
        # print(self.d[start_grad_idx])
        # print(f'ok:{self.log_inverse(self.alpha/self.beta,start_grad_idx)}')
        opt_arr,x_arr,feasible = self.log_inverse(self.alpha/self.beta)

        if feasible:
            print('DONE')
            return opt_arr,x_arr,0

        r = np.max([opt_arr,np.sum(self.l*self.d)*1.1])#np.sum(self.log_inverse(self.alpha/self.beta,start_grad_idx)*self.n*self.d)
        # r = 100
        print(f'r:{r}')
        curr_sol = np.Inf
        prev_sol = -np.Inf
        calls = 0
        while np.abs(curr_sol-prev_sol) > self.eps:
            calls += 1
            prev_sol = curr_sol
            _, mult,x_vars = self.L_hat(r)
            curr_sol = self.lam(r)[0]
            r = r - self.tau*(self.alpha+self.beta*mult)
            # print(curr_sol)


        return r, x_vars, calls

    def draw(self,show=True,validate_point=None,lower_x_lim=0,upper_x_lim=100):

        lsols = []
        sols = []
        multipliers = []
        x_vars_l = []
        rng = np.linspace(lower_x_lim,upper_x_lim,100)
        for i in rng:
            sol,lsol,multiplier,x_vars = self.lam(i)
            sols.append(sol)
            lsols.append(lsol)
            multipliers.append(multiplier)
            x_vars_l.append(x_vars)

        if show:
            plt.plot(rng,sols)

            if validate_point is not None:
                for pt in validate_point:
                    plt.scatter(pt[0],pt[1])
            plt.show()
        else:
            return rng[np.argmin(sols)]

# d = np.array([1,2.2,4.2,.1,1])
# a = np.array([1.1,1.2,2.3,1.4,9.5])
# c = np.array([0]*5)
# b = np.array([1]*5)

# alpha = .15
# beta = .1

# l = [2]*5



# tau= 6.66#110.8
# eps = 0.0001


# a = np.array([2,3,5,4,7])
# b = np.array([1,1,1,1,1])
# c = np.array([1,1,1,1,1])
# d = np.array([1,1,1,1,1])
# l = np.array([0,0,0,0,0])
# alpha = .2
# beta = 1
# n = len(a)

# pr = problem(d,a,b,c,l,alpha,beta,eps,tau)
# r,x=pr.solve()
# pr.draw([r,pr.lam(r)[0]])