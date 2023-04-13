import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.cm as cm
from itertools import cycle

data=pandas.read_excel("./results/linear/AuditResults.xlsx",0).fillna("")

def all_plot(data,descs,ax):
    x="Attack AUC"
    y="Test Acc (%)"
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    styles=cycle(['h-','^-','+-','*-','d-','x-','o-','.-','s-'])
    for desc,style in zip(descs,styles):
        d1=data[data["Description"]==desc].sort_values(by=x)
        d1.plot(x=x,y=y,label=desc,style=style,ax=ax,markersize=12)

# def plot_fc(data,x,y,ax):
#     ax.set_xlabel("Attack AUC")
#     ax.set_ylabel("Test Acc (%)")
#     styles=cycle(['h-','^-','+-','*-','d-','x-','o-','.-','s-'])
#     for xv,yv,style in zip(x,y,styles):
#         data=data.sort_values(by=xv)
#         y_name=yv.split()[0]
#         data.plot(x=xv,y=yv,label=y_name,style=style,ax=ax,markersize=12)

def plot_fc(data,x,y,ax):
    ax.set_xlabel("Attack AUC")
    ax.set_ylabel("Test Acc (%)")
    styles=cycle(['h','^','+','*','d','x','o','.','s'])
    colors = cm.rainbow(np.linspace(0, 1, 8))
    for xv,yv,style in zip(x,y,styles):
        y_name=yv.split()[0]
        data.plot(kind="scatter",x=xv,y=yv,marker=style,label=y_name,ax=ax,s=200,c=colors)

def dual_y_plot(data,desc,ax):
    x="Param"
    y1="Attack AUC"
    y2="Test Acc (%)"
    d1=data[data["Description"]==desc].sort_values(by=x)
    ax1=d1.plot(x=x,y=y1,style=".-",title=desc,ax=ax)
    ax2=d1.plot(x=x,y=y2,style=".-",secondary_y=True,ax=ax)
    ax1.set_ylabel(y1)
    ax2.set_ylabel(y2)
    ax.set_xlabel('Sparsity')

fig1,axs1=plt.subplots(1)
descs=['default','Partial DPSGD','End Mag fc1', 'End Mag fc2', 'End Mag fc1 + fc2',
       'Partial Noise','Partial DPSGD fc1','Partial DPSGD fc2','Partial DPSGD fc1 + fc2']
all_plot(data,descs,axs1)

fig2,axs2=plt.subplots(2,4)
for i,desc in enumerate(descs[2:]):
    dual_y_plot(data,desc,axs2[i//4,i%4])

# fig3,axs3=plt.subplots(1)
# axs3.set_title('First Conv Magnitude Prune 0.02 vs Eps3')
# plot_fc(data2,['ema Attack AUC','last Attack AUC','polyak Attack AUC'],['ema Test Acc (%)','last Test Acc (%)','polyak Test Acc (%)'],axs3)
# d1=data[data["Description"]=='Eps3'].sort_values(by='Attack AUC')
# d1.plot(kind="scatter",x='Attack AUC',y='Test Acc (%)',label='Eps3',marker='o',ax=axs3,s=200,c='black')

plt.show()