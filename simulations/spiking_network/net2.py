"""Slower-dynamics version of the network so bursts last about a second and recur every few seconds.
tau_d and U control synaptic depression; w the coupling. Model assumptions only."""
import numpy as np
from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Network, defaultclock, ms, mV, second, seed as bseed, prefs)
prefs.codegen.target = 'numpy'
def simulate(g_ee=1.0, g_ie=1.0, drive=0.0, T=60.0, seed=0, n_e=240, n_i=60, sigma=3.0, dt_ms=1.0,
             tau_d=4000.0, U=0.04, w_mV=1.4, p=0.12, base=10.0):
    bseed(seed); np.random.seed(seed); defaultclock.dt = dt_ms*ms
    eqs='''
    dv/dt = (-(v - vrest) + drive_mV)/tau + sigma_mV*sqrt(2/tau)*xi : volt (unless refractory)
    dx/dt = (1 - x)/tau_d : 1
    drive_mV : volt (constant)
    '''
    ns={'vrest':-65*mV,'tau':20*ms,'tau_d':tau_d*ms,'sigma_mV':sigma*mV,'U':U}
    G=NeuronGroup(n_e+n_i,eqs,threshold='v > -50*mV',reset='v = -65*mV; x -= U*x',refractory=3*ms,method='euler',namespace=ns)
    G.v='-65*mV + rand()*12*mV'; G.x=1; G.drive_mV=(base+drive)*mV
    E,I=G[:n_e],G[n_e:]; w=w_mV*mV
    see=Synapses(E,E,on_pre='v_post += gee*w*x_pre',namespace={'gee':g_ee,'w':w}); see.connect(p=p)
    sei=Synapses(E,I,on_pre='v_post += w*x_pre*1.5',namespace={'w':w}); sei.connect(p=p)
    sie=Synapses(I,E,on_pre='v_post -= gie*w*3',namespace={'gie':g_ie,'w':w}); sie.connect(p=p)
    sii=Synapses(I,I,on_pre='v_post -= w*3',namespace={'w':w}); sii.connect(p=p)
    M=SpikeMonitor(G); Network(G,see,sei,sie,sii,M).run(T*second)
    return np.asarray(M.t/second), np.asarray(M.i), n_e+n_i
