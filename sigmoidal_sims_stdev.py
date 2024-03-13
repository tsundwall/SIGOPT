import bilevel_analytical as bi
import scipy
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mtick

num_zeros_arr = []
sols = []
np.random.seed(121)
for stdev in np.linspace(.05,3,50):#np.linspace(1,100,10):
    # sols_inner = []
    # for c_len in [10]:#np.logspace(1,4,30,base=10):

        # print(c_len)

    c = np.random.normal(8.4,stdev, size=(int(10),))
    print(c)

    a = 1#a_i
    b = 1#a_i
    alpha = .082
    beta = 1

    pr = bi.problem(a,b,c,alpha,beta)
    t = time.time()
    sol = pr.solve()

    num_zeros = np.sum(np.array(sol['Optimal x']) == 0)
    # print(num_zeros)
    sols.append([sol['Optimal x'],stdev])
    num_zeros_arr.append(num_zeros)



fig,ax = plt.subplots()
ax2 = ax.twinx()
# ax.
print(np.array(num_zeros_arr)/10)
colors = plt.cm.Reds(np.linspace(0,1,8))
# print(sols[0])
for i,stdev_i in enumerate(sols):

    for j,x_i in enumerate(stdev_i[0]):
    # sols_i = np.array(sols_i).T
        if x_i > 0:
            ax.scatter(stdev_i[1],x_i/np.sum(stdev_i[0])*100,c='b',alpha=0.7)
plotted1 = ax.scatter(stdev_i[1],x_i/np.sum(stdev_i[0])*100,c='b',alpha=0.7,label=r'% of allocation to $x_i$')
plotted2 = ax2.bar(np.linspace(.05,3,50),np.array(num_zeros_arr)*10,alpha=0.5,width=.05,color='b',label=r'% of $x_i$ with no allocation')

# formatter = mtick.FormatStrFormatter('$%.0f')
# plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
# plotted = plotted1+plotted2
# labs = [plotted1.get_label() for l in plotted]
fig.legend(loc='upper right', bbox_to_anchor=(0.5,1),bbox_transform=ax.transAxes)
# plt.title('Execution time, different problem sizes')
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'distribution of $x$ allocation')
ax2.set_ylabel(r'% of $x_i$ with no allocation')
# cbaxes = inset_axes(ax, width="3%", height="30%", loc=2, borderpad=2)
# sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=.2, vmax=.25))

# cbar = plt.colorbar(sm,cax=cbaxes,label=r'  $\alpha$',pad=0.8)
# cbar.ax.set_ylabel(r'$\alpha$',rotation=0,labelpad=10)
# cbar.set_ticks([0.2,0.225,0.25])
# ax2.legend(loc=0)
# ax.legend(loc=0)

plt.show()
