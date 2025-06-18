from posydon.popsyn.synthetic_population import Rates
import numpy as np
import h5py
import copy
import corner
import sys
from astropy.cosmology import Planck15
from astropy import units as u

rates = Rates(filename=sys.argv[1], transient_name='BBH', SFH_identifier='IllustrisTNG')
pop = rates.population
thetas = np.array([pop[key].to_numpy() for key in ['S1_mass', 'S2_mass', 'chi_eff']]).T
z = rates.z_events.to_numpy()
weights = rates.weights.to_numpy()

thetas_new = [ ]
for j in range(thetas.shape[-1]):
    this_theta = thetas[:,j]
    this_theta = this_theta[None,:] * np.ones(z.shape[-1])[:,None]
    thetas_new.append(this_theta.flatten()[:,None])
thetas = np.concatenate(tuple(thetas_new), axis = -1)
z = z.flatten()
weights = weights.flatten()
print(thetas.shape, z.shape)

arg = np.isnan(z)
z[arg] = 0.
weights[arg] = 0.

n_cbc_per_pop = 1000000
indices = np.random.choice(len(z), p = weights/weights.sum(), size = n_cbc_per_pop )

thetas = np.concatenate((thetas[indices,:], z[indices, None]), axis = -1)

thetas_copy = copy.deepcopy(thetas)
thetas[:,2] = thetas_copy[:,-1]
thetas[:,-1] = thetas_copy[:, 2]
arg = thetas[:,1]>thetas[:,0]
thetas[:,0][arg] = thetas_copy[:,1][arg]
thetas[:,1][arg] = thetas_copy[:,0][arg]

highs = np.quantile(thetas, 0.95, axis = 0)*1.3
lows = np.min(thetas, axis = 0)

limits = [(low, high) for low, high in zip(lows, highs)]
print(np.where(np.isnan(thetas)))

with h5py.File(f"../__run__/bbh_params_{sys.argv[1][:-3]}_prod.h5", "w") as hf:
    hf.create_dataset("parameters", data = thetas)

fig = corner.corner(thetas, hist_kwargs = {"density":True}, smooth = 1.8, labels = [r'$m_1$', r'$m_2$', r'$z$',r'$\chi_{eff}$'], plot_datapoints = True, range = limits)
fig.savefig(f"{sys.argv[1][:-3]}_params_rateweighted.png")

