import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.cm as cm
import itertools

data=pandas.read_excel("./results/new/AuditResults.xlsx",0).fillna("")

def plot(data,desc,x,y,ax,style=".-"):
    d1=data[data["Description"].str.contains(desc,regex=False)].sort_values(by=x)
    d1.plot(x=x,y=y,label=desc,style=style,ax=ax)

fig1,axs1=plt.subplots(1)
axs1.set_xlabel("Audit Acc")
axs1.set_ylabel("Test Acc (%)")
plot(data,"Augmentation","Attack AUC","Test Acc (%)",axs1)
plot(data,"SOTA","Attack AUC","Test Acc (%)",axs1)

plt.show()