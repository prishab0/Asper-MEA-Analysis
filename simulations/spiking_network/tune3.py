import sys, numpy as np, time
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from net2 import simulate
def stats(t,n,T):
    pr,_=np.histogram(t,bins=np.arange(0,T+0.01,0.01)); pr=gaussian_filter1d(pr/0.01/n,5)
    base=np.median(pr); pk,pp=find_peaks(pr,prominence=max(3*base,4),distance=100)
    if len(pk)<3: return len(t)/T/n,len(pk),0,0,0
    wd=[]
    for p in pk:
        h=pr[p]; half=base+(h-base)/2; lo=p
        while lo>0 and pr[lo]>half: lo-=1
        hi=p
        while hi<len(pr)-1 and pr[hi]>half: hi+=1
        wd.append((hi-lo)*0.01)
    return len(t)/T/n,len(pk),np.mean(wd),np.mean(np.diff(pk)*0.01),np.mean(wd)*len(pk)/T
if __name__=="__main__":
    pass
