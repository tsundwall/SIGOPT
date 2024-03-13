import bilevel_analytical as bi
import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sols = []
for mean in np.linspace(.2,.25,8):#np.linspace(1,100,10):
    print(mean)
    sols_inner = []
    for c_len in np.logspace(1,4,30,base=10):

        # print(c_len)
        c = np.random.normal(9,2, size=(int(c_len),))


        a = 1#a_i
        b = 1#a_i
        alpha = mean#.2
        beta = 1

        pr = bi.problem(a,b,c,alpha,beta)
        t = time.time()
        sol = pr.solve()
        sols_inner.append([c_len,time.time()-t,mean])
    sols.append(sols_inner)

fig = plt.figure()
ax = plt.subplot(111)
colors = plt.cm.Reds(np.linspace(0,1,8))
plt.xscale('log')
for i,sols_i in enumerate(sols):
    sols_i = np.array(sols_i).T

    plt.scatter(sols_i[0],sols_i[1],c=colors[i])
# plt.title('Execution time, different problem sizes')
plt.xlabel(r'$n$')
plt.ylabel('execution time (secs)')
cbaxes = inset_axes(ax, width="3%", height="30%", loc=2, borderpad=2)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=.2, vmax=.25))

cbar = plt.colorbar(sm,cax=cbaxes,label=r'  $\alpha$',pad=0.8)
cbar.ax.set_ylabel(r'$\alpha$',rotation=0,labelpad=10)
cbar.set_ticks([0.2,0.225,0.25])

plt.show()
