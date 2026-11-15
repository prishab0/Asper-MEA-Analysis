import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8.5})
feats=["Firing rate","Shape 1D","Shape 2D","Shape 3D"]
W={50:[(.633,.411,.845),(.683,.464,.892),(1,1,1),(1,1,1)],
   100:[(.633,.398,.845),(.628,.391,.847),(1,1,1),(1,1,1)],
   200:[(.633,.398,.847),(.683,.464,.894),(1,1,1),(1,1,1)]}
X={50:[(.5,.5,.5),(.833,.5,1),(.85,.7,1),(.95,.833,1)],
   100:[(.5,.5,.5),(.833,.5,1),(.85,.7,1),(.95,.833,1)],
   200:[(.5,.5,.5),(1,1,1),(.85,.7,1),(.95,.833,1)]}
mk={50:("o","k"),100:("s","#555"),200:("^","#999")}
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True,dpi=200)
for ax,(D,t) in zip(axs,((W,"a. Day 18 within plate (19 wells, pooled rate range)"),(X,"b. Train day 18, test day 17 (13 wells)"))):
    for j,ms in enumerate((50,100,200)):
        for i,(v,lo,hi) in enumerate(D[ms]):
            y=i+(j-1)*0.2
            ax.plot([lo,hi],[y,y],color=mk[ms][1],lw=1.1,zorder=2)
            ax.scatter(v,y,marker=mk[ms][0],s=20,color=mk[ms][1],zorder=3,label=f"minSburst {ms}" if i==0 else None)
    ax.axvline(.5,color="#aaa",lw=.7,ls="--",zorder=1)
    ax.set_yticks(range(4)); ax.set_yticklabels(feats); ax.set_xlim(.3,1.05); ax.invert_yaxis()
    ax.set_xlabel("Balanced accuracy"); ax.set_title(t,fontsize=8.5,loc="left")
    ax.spines[["top","right"]].set_visible(False)
axs[1].legend(frameon=False,fontsize=7.5,loc="upper left")
fig.tight_layout(); fig.savefig("fig8.png")
