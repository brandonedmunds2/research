# 0.9687730738

import matplotlib.pyplot as plt
import pandas
import numpy as np
import matplotlib.cm as cm
import itertools

# lin=pandas.read_excel("AuditPrivacy.xlsx",1).fillna("")
# lin=lin[~lin["Pruning Scope"].str.contains("layer 1")]

# conv=pandas.read_excel("AuditPrivacy.xlsx",2).fillna("")
# conv=conv[conv["Reference Models"]!=50]
# conv=conv[conv["Epochs"]==100]

# data3=pandas.read_excel("AuditPrivacy.xlsx",3).fillna("")

data4=pandas.read_excel("AuditPrivacy.xlsx",4).fillna("")
data4=data4[data4["Prune Epochs"]!=1]

def dual_y_plot(data,desc,x,y,ax,style=".-"):
    d1=data[data["Description"].str.contains(desc,regex=False)].sort_values(by=x)
    d1.plot(x=x,y=y,label=desc,style=style,ax=ax)

def y_plot(data,desc,x,y1,y2,ax,scope=""):
    d1=data[data["Description"].str.contains(desc,regex=False)].sort_values(by=x)
    d1=d1[d1["Pruning Scope"].str.contains(scope)]
    if(scope != ""):
        scope=" "+scope
    d1.plot(x=x,y=y1,style=".-",title=desc+scope,ax=ax)
    d1.plot(x=x,y=y2,style=".-",secondary_y=True,ax=ax)

# def y_plot_r(data,desc,x,y1,y2,ax,r):
#     d1=data[data["Description"].str.contains(desc,regex=False)].sort_values(by=x)
#     d1=d1[d1["Pruning Rate 2"] == r]
#     r="; mag prune = "+str(r)
#     d1.plot(x=x,y=y1,style=".-",title=desc+r,ax=ax)
#     d1.plot(x=x,y=y2,style=".-",secondary_y=True,ax=ax)

# def b_plot(data,descs,x,y,ax,target_tst_acc=73):
#     markers = itertools.cycle(('x','X',',', '+', '.', 'o', '*')) 
#     colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, len(descs)+2)))
#     for desc in descs:
#         d1=data[data["Description"].str.contains(desc)]
#         d2=d1[d1["Pruning Scope"].str.contains("conv")]
#         sc1=""
#         if(len(d2)!=0):
#             d1=data[data["Pruning Scope"].str.contains("linear")]
#             sc1=" linear"
#             sc2=" conv"
#             l2=d2[d2[x]>=target_tst_acc]
#             s2=d2[d2[x]<=target_tst_acc]
#             s2=s2[s2[x]==s2[x].max()]
#             l2=l2[l2[x]==l2[x].min()]
#             sa2=(s2[x]-target_tst_acc).abs()
#             la2=(l2[x]-target_tst_acc).abs()
#             if(la2.max()>sa2.max()):
#                 b2=s2
#             else:
#                 b2=l2
#             b2.plot.scatter(x=x,y=y,color=[next(colors)],ax=ax,label=desc+sc2,title=y,marker=next(markers),s=50)
#         l=d1[d1[x]>=target_tst_acc]
#         s=d1[d1[x]<=target_tst_acc]
#         s=s[s[x]==s[x].max()]
#         l=l[l[x]==l[x].min()]
#         sa=(s[x]-target_tst_acc).abs()
#         la=(l[x]-target_tst_acc).abs()
#         if(la.max()>sa.max()):
#             b=s
#         else:
#             b=l
#         b.plot.scatter(x=x,y=y,color=[next(colors)],ax=ax,label=desc+sc1,title=y,marker=next(markers),s=50)

# def b_plot_r(data_s,descs_s,datar,descr,rps,x,y,ax,tot,target_tst_acc=73):
#     markers = itertools.cycle(('x','X',',', '+', '.', 'o', '*')) 
#     colors = itertools.cycle(cm.rainbow(np.linspace(0, 1, tot)))
#     for i in range(len(data_s)):
#         for desc in descs_s[i]:
#             d1=data_s[i][data_s[i]["Description"].str.contains(desc,regex=False)]
#             l=d1[d1[x]>=target_tst_acc]
#             s=d1[d1[x]<=target_tst_acc]
#             s=s[s[x]==s[x].max()]
#             l=l[l[x]==l[x].min()]
#             sa=(s[x]-target_tst_acc).abs()
#             la=(l[x]-target_tst_acc).abs()
#             if(len(sa)!=0 and (len(la)==0 or la.max()>sa.max())):
#                 b=s
#             else:
#                 b=l
#             sc1="; " + str(b["Pruning Rate"].values[0])
#             b.plot.scatter(x=x,y=y,color=[next(colors)],ax=ax,label=desc+sc1,title=y,marker=next(markers),s=50)
#     for desc,rp in zip(descr,rps):
#         d1=datar[datar["Description"].str.contains(desc,regex=False)]
#         d1=d1[d1["Pruning Rate 2"] == rp]
#         l=d1[d1[x]>=target_tst_acc]
#         s=d1[d1[x]<=target_tst_acc]
#         s=s[s[x]==s[x].max()]
#         l=l[l[x]==l[x].min()]
#         sa=(s[x]-target_tst_acc).abs()
#         la=(l[x]-target_tst_acc).abs()
#         if(la.max()>sa.max()):
#             b=s
#         else:
#             b=l
#         sc1="; " + str(b["Pruning Rate"].values[0])
#         sc1+="; mag prune = "+str(rp)
#         b.plot.scatter(x=x,y=y,color=[next(colors)],ax=ax,label=desc+sc1,title=y,marker=next(markers),s=50)
#     ax.legend(loc='upper center',bbox_to_anchor=(0.5,2))

# def net_plot(net):
#     if(net=="Linear"):
#         data=lin
#     else:
#         data=conv
#     fig1,axs1=plt.subplots(2,5)
#     fig1.suptitle(net)
#     if(net != "Linear"):
#         sc1="linear"
#     else:
#         sc1=""
#     y_plot(data,"end magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,0],sc1)
#     y_plot(data,"end magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,0],sc1)
#     y_plot(data,"end random weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,1])
#     y_plot(data,"end random weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,1])
#     y_plot(data,"mid magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,2])
#     y_plot(data,"mid magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,2])
#     y_plot(data,"mid random weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,3])
#     y_plot(data,"mid random weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,3])
#     y_plot(data,"per sample random gradient prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,4])
#     y_plot(data,"per sample random gradient prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,4])
#     if(net != "Linear"):
#         fig3,axs3=plt.subplots(1,2)
#         fig3.suptitle(net)
#         y_plot(data,"end magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs3[0],"conv")
#         y_plot(data,"end magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs3[1],"conv")

#     fig2,axs2=plt.subplots(1,2)
#     fig2.suptitle(net)
#     b_plot(data,["end magnitude weight prune","end random weight prune","mid magnitude weight prune","mid random weight prune","per sample random gradient prune","opacus","default"],"Test Acc (%)","ROC AUC",axs2[0])
#     b_plot(data,["end magnitude weight prune","end random weight prune","mid magnitude weight prune","mid random weight prune","per sample random gradient prune","opacus","default"],"Test Acc (%)","Audit Acc",axs2[1])


fig1,axs1=plt.subplots(2,4)
y_plot(data4,"linear + magnitude","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,0])
y_plot(data4,"linear + magnitude","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,0])
y_plot(data4,"linear + snip","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,1])
y_plot(data4,"linear + snip","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,1])
y_plot(data4,"linear + grasp","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,2])
y_plot(data4,"linear + grasp","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,2])
y_plot(data4,"linear + synflow","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,3])
y_plot(data4,"linear + synflow","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,3])

fig2,axs2=plt.subplots(1,2)
axs2[0].set_ylabel("ROC AUC")
axs2[1].set_ylabel("Audit Acc")
axs2[0].set_xlabel("Test Acc (%)")
axs2[1].set_xlabel("Test Acc (%)")
dual_y_plot(data4,"linear + magnitude","Test Acc (%)","ROC AUC",axs2[0])
dual_y_plot(data4,"linear + magnitude","Test Acc (%)","Audit Acc",axs2[1])
dual_y_plot(data4,"linear + snip","Test Acc (%)","ROC AUC",axs2[0])
dual_y_plot(data4,"linear + snip","Test Acc (%)","Audit Acc",axs2[1])
dual_y_plot(data4,"linear + grasp","Test Acc (%)","ROC AUC",axs2[0])
dual_y_plot(data4,"linear + grasp","Test Acc (%)","Audit Acc",axs2[1])
dual_y_plot(data4,"linear + synflow","Test Acc (%)","ROC AUC",axs2[0])
dual_y_plot(data4,"linear + synflow","Test Acc (%)","Audit Acc",axs2[1])
dual_y_plot(data4,"linear + default","Test Acc (%)","ROC AUC",axs2[0],style="*")
dual_y_plot(data4,"linear + default","Test Acc (%)","Audit Acc",axs2[1],style="*")
dual_y_plot(data4,"linear + opacus","Test Acc (%)","ROC AUC",axs2[0])
dual_y_plot(data4,"linear + opacus","Test Acc (%)","Audit Acc",axs2[1])

# net_plot("Linear")
# net_plot("CNN")

# fig1,axs1=plt.subplots(2,2)
# fig1.suptitle("Linear")
# y_plot(data3,"linear + init random weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,0])
# y_plot(data3,"linear + init random weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,0])
# y_plot(data3,"linear + init magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs1[0,1])
# y_plot(data3,"linear + init magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs1[1,1])

# fig4,axs4=plt.subplots(2,3)
# y_plot(data3,"linear + magnitude grad prune","Pruning Rate","Test Acc (%)","ROC AUC",axs4[0,0])
# y_plot(data3,"linear + magnitude grad prune","Pruning Rate","Test Acc (%)","Audit Acc",axs4[1,0])
# y_plot(data3,"linear + reverse magnitude grad prune","Pruning Rate","Test Acc (%)","ROC AUC",axs4[0,1])
# y_plot(data3,"linear + reverse magnitude grad prune","Pruning Rate","Test Acc (%)","Audit Acc",axs4[1,1])
# y_plot(data3,"linear + mid magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs4[0,2])
# y_plot(data3,"linear + mid magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs4[1,2])

# fig2,axs2=plt.subplots(2,2)
# y_plot_r(data3,"linear + end + mid magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs2[0,0],0.003)
# y_plot_r(data3,"linear + end + mid magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs2[1,0],0.003)
# y_plot_r(data3,"linear + end + mid magnitude weight prune","Pruning Rate","Test Acc (%)","ROC AUC",axs2[0,1],0.001)
# y_plot_r(data3,"linear + end + mid magnitude weight prune","Pruning Rate","Test Acc (%)","Audit Acc",axs2[1,1],0.001)

# fig3,axs3=plt.subplots(1,2)
# fig2.suptitle(net)
# l2=["linear + init random weight prune","linear + init magnitude weight prune","linear + magnitude grad prune","linear + reverse magnitude grad prune","linear + mid magnitude weight prune","conv + linear"]
# b_plot_r([lin,data3],[["end magnitude weight prune","end random weight prune","mid magnitude weight prune","mid random weight prune","per sample random gradient prune","opacus","default"],l2],data3,["linear + end + mid magnitude weight prune","linear + end + mid magnitude weight prune"],[0.003,0.001],"Test Acc (%)","ROC AUC",axs3[0],15)
# b_plot_r([lin,data3],[["end magnitude weight prune","end random weight prune","mid magnitude weight prune","mid random weight prune","per sample random gradient prune","opacus","default"],l2],data3,["linear + end + mid magnitude weight prune","linear + end + mid magnitude weight prune"],[0.003,0.001],"Test Acc (%)","Audit Acc",axs3[1],15)

plt.show()