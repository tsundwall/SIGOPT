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

    def lam(self,r_t):
        sol,multipliers,x_vars = self.L_hat(r_t)
        return self.alpha*r_t + self.beta*(-sol),sol,multipliers,x_vars


    def L_hat(self,r):

        # n = 5
        # A = -np.array([[1,.2,.2,.1,1]])
        # b = -np.array([m])
        # c = np.array([1.1,1.2,2.3,1.4,9.5])

        x = cp.Variable(self.n)

        constraints = [-self.d @ x == -r]

        for i,l_i in enumerate(self.l):

            constraints += [x[i] >= l_i]

        prob = cp.Problem(cp.Minimize(-self.a.T@cp.log(cp.multiply(self.b.T,x)+self.c)+2),
                        constraints)

        sol = prob.solve()

        multipliers = constraints[0].dual_value

        return -sol,multipliers,x.value

    def solve(self):

        r = 20
        curr_sol = np.Inf
        prev_sol = -np.Inf

        while np.abs(curr_sol-prev_sol) > self.eps:
            prev_sol = curr_sol
            _, mult,x_vars = self.L_hat(r)
            curr_sol = self.lam(r)[0]
            r = r - self.tau*(self.alpha+self.beta*mult)
            # print(curr_sol)


        return r, x_vars

    def draw(self,validate_point=None):

        lsols = []
        sols = []
        multipliers = []
        x_vars_l = []

        for i in range(1,53):
            sol,lsol,multiplier,x_vars = self.lam(i)
            sols.append(sol)
            lsols.append(lsol)
            multipliers.append(multiplier)
            x_vars_l.append(x_vars)


        plt.plot(range(1,53),sols)

        if validate_point is not None:
            plt.scatter(validate_point[0],validate_point[1])
        plt.show()

d = np.array([1,2.2,4.2,.1,1])
a = np.array([1.1,1.2,2.3,1.4,9.5])
c = np.array([0]*5)
b = np.array([1]*5)

l = [2]*5

tau= 6.66#110.8
eps = 0.001

alpha = .15
beta = .1
pr = problem(d,a,b,c,l,alpha,beta,eps,tau)
r,x=pr.solve()
# pr.draw([r,pr.lam(r)[0]])