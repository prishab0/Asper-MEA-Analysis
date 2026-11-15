"""Small excitatory/inhibitory spiking network with depressing E synapses.
Population bursts emerge from recurrent excitation plus synaptic depression.
Everything here is a model assumption, not data."""
import numpy as np
from brian2 import (NeuronGroup, Synapses, SpikeMonitor, Network, defaultclock,
                    ms, mV, second, seed as bseed, prefs)
prefs.codegen.target = 'numpy'

def simulate(g_ee=1.0, g_ie=1.0, drive=0.0, T=30.0, seed=0, n_e=320, n_i=80,
             sigma=3.0, dt_ms=0.5):
    """g_ee: E->E gain, g_ie: I->E gain, drive: extra background (mV). Returns (t_s, id, n_e)."""
    bseed(seed); np.random.seed(seed)
    defaultclock.dt = dt_ms * ms
    eqs = '''
    dv/dt = (-(v - vrest) + drive_mV)/tau + sigma_mV*sqrt(2/tau)*xi : volt (unless refractory)
    dx/dt = (1 - x)/tau_d : 1
    drive_mV : volt (constant)
    '''
    ns = {'vrest': -65*mV, 'tau': 20*ms, 'tau_d': 600*ms, 'sigma_mV': sigma*mV}
    G = NeuronGroup(n_e + n_i, eqs, threshold='v > -50*mV', reset='v = -65*mV; x -= 0.25*x',
                    refractory=3*ms, method='euler', namespace=ns)
    G.v = '-65*mV + rand()*12*mV'
    G.x = 1
    G.drive_mV = (10.0 + drive) * mV
    E, I = G[:n_e], G[n_e:]
    p = 0.12
    w = 1.4 * mV
    see = Synapses(E, E, on_pre='v_post += gee*w*x_pre', namespace={'gee': g_ee, 'w': w}); see.connect(p=p)
    sei = Synapses(E, I, on_pre='v_post += w*x_pre*1.5', namespace={'w': w}); sei.connect(p=p)
    sie = Synapses(I, E, on_pre='v_post -= gie*w*3', namespace={'gie': g_ie, 'w': w}); sie.connect(p=p)
    sii = Synapses(I, I, on_pre='v_post -= w*3', namespace={'w': w}); sii.connect(p=p)
    M = SpikeMonitor(G)
    net = Network(G, see, sei, sie, sii, M)
    net.run(T * second)
    return np.asarray(M.t / second), np.asarray(M.i), n_e + n_i
