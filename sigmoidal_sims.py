import bilevel_analytical as bi
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sols_outer_outer = []
sols_outer = []

c = np.random.normal(6,3, size=(5,))

for a_i in np.linspace(.5,5,20):
    sols = []
    for alph in np.linspace(.01,1.25,300):

        a = a_i
        b = 1#a_i
        alpha = alph
        beta = 1

        pr = bi.problem(a,b,c,alpha,beta)
        sol = pr.solve()
        sols.append([sol['Optimal r'],alph])
    sols_outer.append(sols)

sols_outer_outer.append(sols_outer)

ax = plt.subplot(111)

colors = plt.cm.Blues(np.linspace(0,1,len(sols_outer)))
for i,sim in enumerate(sols_outer):
    sim = np.array(sim).T
    plt.title(r'Optimal resource level for varying $\alpha$ and $a$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$r^*$')
    plt.plot(sim[1],sim[0],label=i,color=colors[i])

cbaxes = inset_axes(ax, width="3%", height="30%", loc=1, borderpad=3)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=.5, vmax=5))

cbar = plt.colorbar(sm,cax=cbaxes)
cbar.ax.set_ylabel(r'$a$',rotation=0,labelpad=10)
cbar.ax.yaxis.set_label_position('left')
cbar.ax.yaxis.set_ticks_position('left')

cbar.set_ticks([0.5,2.75,5])

plt.show()