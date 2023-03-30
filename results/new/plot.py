import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.cm as cm
import itertools

data=pandas.read_excel("./results/new/AuditResults.xlsx",0).fillna("")

def all_plot(data,ax):
    x="Attack AUC"
    y="Test Acc (%)"
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    descs=data["Description"].unique()
    styles=['h-','^-','+-','*-','d-','x-','o-','.-','s-']
    for desc,style in zip(descs,styles):
        d1=data[data["Description"]==desc].sort_values(by=x)
        d1.plot(x=x,y=y,label=desc,style=style,ax=ax,markersize=12)

def dual_y_plot(data,desc,ax):
    x="Param"
    y1="Attack AUC"
    y2="Test Acc (%)"
    d1=data[data["Description"]==desc].sort_values(by=x)
    ax1=d1.plot(x=x,y=y1,style=".-",title=desc,ax=ax)
    ax2=d1.plot(x=x,y=y2,style=".-",secondary_y=True,ax=ax)
    ax1.set_ylabel(y1)
    ax2.set_ylabel(y2)
    ax.set_xlabel('Density Fraction')

fig1,axs1=plt.subplots(1)
all_plot(data,axs1)

fig2,axs2=plt.subplots(2,2)
dual_y_plot(data,"Magnitude Prune",axs2[0,0])
dual_y_plot(data,"Linear Magnitude Prune",axs2[0,1])
dual_y_plot(data,"Conv Magnitude Prune",axs2[1,0])
dual_y_plot(data,"Random Prune",axs2[1,1])

fig3,axs3=plt.subplots(1)
dual_y_plot(data,"First Conv Magnitude Prune",axs3)

plt.show()