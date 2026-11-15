import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open('wells.json'))
names={'rescue':'Rescue (healthy parameters)','supp_drive':'Suppression: lower drive','supp_gee':'Suppression: weaker excitation','supp_gie':'Suppression: stronger inhibition','disease':'Untreated disease'}
cols={'rescue':'#1f77b4','supp_drive':'#d62728','supp_gee':'#ff7f0e','supp_gie':'#2ca02c','disease':'#7f7f7f'}
fig,ax=plt.subplots(1,2,figsize=(11,4))
x=np.arange(-250,250,5)
for a in names:
    W=np.array([r['wave'] for r in R if r['arm']==a and 'wave' in r])
    m=W.mean(0); s=W.std(0)
    ax[0].plot(x,m,color=cols[a],label=names[a],lw=2 if a!='disease' else 1.2,ls='-' if a!='disease' else '--'); ax[0].fill_between(x,m-s,m+s,color=cols[a],alpha=0.12)
ax[0].set_xlabel('time from burst peak (ms)'); ax[0].set_ylabel('population rate / peak'); ax[0].set_title('a. Mean burst shape per arm (24 wells each)',fontsize=10); ax[0].legend(fontsize=7.5,frameon=False)
labs=['lower drive','weaker excitation','stronger inhibition']
rate=[0.530,0.530,0.474]; nb=[0.998,0.863,0.528]; sh=[1.000,0.905,0.477]
xx=np.arange(3); w=0.26
ax[1].bar(xx-w,rate,w,label='mean firing rate',color='#bbbbbb'); ax[1].bar(xx,nb,w,label='burst count',color='#8da0cb'); ax[1].bar(xx+w,sh,w,label='burst shape (3 features)',color='#1f1f1f')
ax[1].axhline(0.5,color='k',ls=':',lw=1); ax[1].set_xticks(xx); ax[1].set_xticklabels(labs,fontsize=8.5); ax[1].set_ylim(0,1.05); ax[1].set_ylabel('leave-one-well-out AUC'); ax[1].set_title('b. Rescue vs rate-matched suppression',fontsize=10); ax[1].legend(fontsize=8,frameon=False,loc='lower left')
for a in ax: 
    for s in ('top','right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig('sim_fig.png',dpi=170)
