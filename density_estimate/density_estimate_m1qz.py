import numpy as np
import h5py
import pickle
import corner

import os
import sys
source = os.getcwd()+'/../src/sbi/'
sys.path.append(source)

import torch
from utils import set_device
from flows.flow import NormalizingFlow
from trainers.train_flows import train

from astropy.cosmology import Planck15
from astropy import units as u

filename = sys.argv[1]
label = filename[filename.index('s_')+2:filename.index(".h5")]
print(label)

with h5py.File(filename, "r") as hf:
    thetas  = hf["parameters"][()][:,:3]
    arg = np.where(thetas[:,1]>thetas[:,0])[0]
    m1 = thetas[:,0].copy()
    m2 = thetas[:,1].copy()

    thetas[:,0][arg] = m2[arg]
    thetas[:,1][arg] = m1[arg]
    m1, m2 = [ ], [ ]
thetas[:,1] = thetas[:,1]/thetas[:,0]
thetas[:,0] = np.log(thetas[:,0])
n_cbc_per_pop = len(thetas)
print(np.where(thetas[:,1]>=1))
bounds = {'low':set_device([min(thetas[:,i])for i in range(thetas.shape[-1])]), 'high': set_device([max(thetas[:,i]) for i in range(thetas.shape[-1])])}

bounds['low']*=0.7
bounds['high']*=1.3
bounds["high"][1] = 1.0
bounds["low"][2] = 0.0


print(bounds)
if sys.argv[2] == "Train":
    flow = NormalizingFlow("maf", bounds, 3, 0, [256, 256, 256, 256], 14)

    model, history, history_val, best_mse,best_epoch = train(flow, set_device(thetas), set_device(torch.zeros(len(thetas))), train_frac = 0.89, patience = 64, lr = 1e-3, min_lr = 1e-8, num_epochs = 2048, batch_frac = 0.05)

    with open(f'estimator_{label}_bounded_m1qz.pkl', 'wb') as f:
        pickle.dump(model,f)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(2048), history, label = 'training')
    plt.plot(np.arange(2048), history_val, label = 'validation')
    plt.axvline(best_epoch)
    plt.axhline(float(best_mse))
    plt.legend()
    plt.savefig(f'loss_evolution_{label}_m1qz.png')
elif sys.argv[2] == "Eval":
    pass
else:
    raise
nparam = thetas.shape[-1]

bounds = {k:b.cpu().detach().numpy() for k,b in bounds.items()}
num_bins = 30
bounds["high"][0] = 5.0
bounds["high"][2] = 10.0
bounds = [(bounds['low'][i], bounds['high'][i]) for i in range(nparam)]
with open(f'estimator_{label}_bounded_m1qz.pkl', 'rb') as f:
    model = pickle.load(f)

samples = model.sample([n_cbc_per_pop,]).cpu().detach().numpy()


fig = corner.corner(thetas, hist_kwargs = {"density": True}, color = 'orange', smooth = 1.2, bins = num_bins+1, range = bounds, plot_datapoints = False)#, weights = weights/weights.sum())

fig = corner.corner(samples, hist_kwargs = {"density": True}, color = 'blue', smooth=1.2, bins = num_bins+1, range = bounds, fig = fig, plot_datapoints = False, labels = [r'$\log\frac{m_1}{M_{\odot}}$', r'$q$', r'$z$'])#, weights = weights/weights.sum())

axes = np.array(fig.axes).reshape((nparam, nparam))



if sys.argv[3] == 'True':
    with h5py.File("../data/pe-inj-data-o3_m1qz.h5", "r") as hf:
        ps = hf["pe_data/posteriors"][()][:,:3, :]
        print(ps.shape)
        print(hf["pe_data/param"][()])
    
    ps[:, 0, :] = np.log(ps[:, 0, :])
    med = np.quantile(ps, 0.5, axis=2)
    ciu = np.quantile(ps, 0.95,axis=2)
    cil = np.quantile(ps, 0.05, axis=2)
    for i in range(nparam):
        for j in range(i+1):
            if i!=j:
                axes[i,j].errorbar(med[:,j], med[:,i], yerr = [ciu[:,i]-med[:,i]], xerr = [med[:,j]-cil[:,j]], fmt = '.',  linestyle = 'none', alpha =0.5)
            else:
                for this_med in med[:,i]:
                    axes[i,j].axvline(this_med, alpha=0.1)





    fig.savefig(f"../__run__/{label}_run_bounded_we_m1qz.png")

else:
    
    fig.savefig(f"../__run__/{label}_run_bounded_m1qz.png")
 



