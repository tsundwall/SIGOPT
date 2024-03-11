import scipy
import numpy as np
import alg as so


def build_sig_funcs_simple(a,c_arr):
    n = len(c_arr)
    funcs = []
    c_arr.sort()
    for i in range(n):
        funcs.append(so.sigmoidal_function(a,1,c_arr[i]))

    return funcs

class problem:
    def __init__(self,a:float,b:float,c:list[float],alpha:float,beta:float) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.n = len(self.c)
        self.sig_funcs = build_sig_funcs_simple(self.a,self.c)

    def eval_l(self,x):

        l_value = 0
        for i,x_i in enumerate(x):
            l_value += self.sig_funcs[i].evaluate(x_i)

        return l_value

    def is_optimal(self,x:list[float],r:float,k:int,left_side:bool) -> bool:

        l_activation = self.eval_l(x)
        if left_side:
            x_perturbed = self.retreive_x(k+1,r)
            l_perturbed = self.eval_l(x_perturbed)

        else:
            x_perturbed = self.retreive_x(k-1,r)
            l_perturbed = self.eval_l(x_perturbed)

        return l_activation > l_perturbed

    def first_deriv_inv(self,y:float) -> float:
        return np.log((np.sqrt(self.a)*np.sqrt(self.b)*np.sqrt(4*y+self.a*self.b)*np.exp(self.c[0])-self.a*self.b*np.exp(self.c[0])-2*np.exp(self.c[0])*y)/(2*y))/self.b

    def find_theta(self) -> float:
        def deriv_first_var(x):
            return [(self.a*np.exp(x[0]+self.c[0]))/((np.exp(x[0])+np.exp(self.c[0]))**2) - (self.alpha/self.beta),x[0]-x[1]**2-self.c[0]]

        # theta,_ = scipy.optimize.fsolve(deriv_first_var,[2*self.c[0],0.01],xtol=1.49012e-08)
        x_alloc = [0]*self.n
        c = self.c[0]
        y = self.alpha / self.beta

        # theta = np.log((self.a*self.b*np.exp(c)-2*np.exp(c)*y+np.sqrt(-a)*np.sqrt(b)*np.exp(c)*np.sqrt(4*y-self.a*self.b)))/(2*y)
        theta = 2*(c/self.b) - self.first_deriv_inv(-self.alpha/self.beta)
        x_alloc[0] = theta

        return theta-self.c[0],x_alloc

    def retreive_x(self,k,r) -> list[float]:

        x_new = []

        sum_inflections = np.sum(self.c[:k])

        remaining = (r-sum_inflections)/(k)

        for i in range(self.n):
            if i < k:
                x_new.append(self.c[i]+remaining)
            else:
                x_new.append(0)

        return x_new

    def solve(self) -> dict:

        minima = []
        minima_r = []
        minima_x = []
        r = 0
        left_side = True

        if self.alpha/self.beta > (self.a*self.b)/4:

            return {
                'Upper Minimum': 0,
                'Optimal r': 0,
                'Optimal x': [0]*self.n
            }

        theta,x = self.find_theta()

        for k in range(0,self.n):

            r += theta + self.c[k]


            x = self.retreive_x(k+1,r)

            if True:#self.is_optimal(x,r,k+1,left_side):

                lam = self.alpha*r - self.beta*self.eval_l(x)
                left_side = False
            # else:
            # if True:#not left_side:
            #     if len(minima) > 0:
            #         return {
            #         'Upper Minimum': minima[-1],
            #         'Optimal r': minima_r[-1],
            #         'Optimal x': minima_x[-1]
            #         }
            #     else:
            #         return {
            #             'fix'
            #         }
            #     # else:
            #     #     continue


            if len(minima) > 0:


                if lam > minima[-1]:

                    return {
                    'Upper Minimum': minima[-1],
                    'Optimal r': minima_r[-1],
                    'Optimal x': minima_x[-1]
                }
                minima.append(lam)
                minima_r.append(r)
                minima_x.append(x)


            else:
                minima.append(lam)
                minima_r.append(r)
                minima_x.append(x)

        return {
                'Upper Minimum': minima[-1],
                'Optimal r': minima_r[-1],
                'Optimal x': minima_x[-1]
            }

c = [1.68755356, 1.69657806, 1.16548715, 2.15513042, 1.91748372,
       2.97135358, 1.90930468, 2.1212319 , 2.28685834, 2.30011484,
       1.30609675, 1.24225479, 2.0825002 , 2.64996499, 1.14816546,
       1.14374299, 2.95539075, 1.76307915, 2.97550369, 2.40989722,
       2.59114533, 1.76773409, 2.00224763, 2.70317089, 1.20777725,
       2.24525831, 2.89310534, 2.75941663, 2.39219145, 2.35303925]

# c = [1,2,3,4,5,6]
np.random.seed(158)
c = np.random.uniform(low=1, high=3, size=(30,))
c = [1.68755356, 1.69657806, 1.16548715, 2.15513042, 1.91748372]
c.sort()

# sols_outer_outer = []
# for c_i in range(5):
#     print(c_i)
sols_outer = []
# c = np.random.uniform(low=1, high=3, size=(30,))
for a_i in np.linspace(.5,5,20):
    sols = []
    for alph in np.linspace(.05,.249,300):

        a = 1#a_i
        b =a_i
        alpha = alph
        beta = 1

        pr = problem(a,b,c,alpha,beta)
        sol = pr.solve()
        sols.append([sol['Optimal r'],alph])
    sols_outer.append(sols)

# sols_outer_outer.append(sols_outer)

import matplotlib.pyplot as plt
# for outer in sols_outer_outer:

for i,sim in enumerate(sols_outer):
    sim = np.array(sim).T

    plt.plot(sim[1],sim[0],label=i)
plt.legend()
plt.show()

