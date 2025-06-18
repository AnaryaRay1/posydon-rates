import numpy as np
from scipy import special
import pickle
import glob

import torch
#from torchns import NestedSampler, UniformPrior, GaussianPrior

import arviz as az

from astropy.cosmology import Planck15
from astropy import units as u
import yaml
import h5py

import os
import sys

source = os.getcwd() + '/../src/sbi/'
sys.path.append(source)
from utils import set_device

from scipy.interpolate import interp1d

def batched_log_prob(func, x, batch_size):
    nbatches = int(len(x)/batch_size) + 1
    import tqdm
    for i in tqdm.tqdm(range(nbatches)):
        f = func(set_device(x[i*batch_size:min((i+1)*batch_size, len(x)),:])).cpu().detach().numpy()
        if i==0:
            out = f.copy()
        else:
            out = np.append(out,f )
        f = [ ]
        torch.cuda.empty_cache()
    return set_device(out)

def load_pe_inj(filename, ignore = None):
    data = az.from_netcdf(filename)
    print(f"data file {filename} loaded")
    if ignore is not None:
        sel = np.zeros(data.pe_data["event"].values.shape, dtype=bool)
        for gw in ignore:
            sel += data.pe_data["event"] == gw
        sel = ~sel
        pedict = {k: np.asarray(data.pe_data.posteriors.sel(param=k).values[sel]) for k in data.pe_data.param.values}
    else:
        pedict = {k: np.asarray(data.pe_data.posteriors.sel(param=k).values) for k in data.pe_data.param.values}

    injdict = {k: np.asarray(data.inj_data.injections.sel(param=k).values) for k in data.inj_data.param.values}

    param_names = list(data.pe_data.param.values)

    total_inj = data.inj_data.attrs["total_generated"]
    obs_time = data.inj_data.attrs["analysis_time"]
    nObs = data.pe_data.posteriors.shape[0]

    constants = {"total_inj": total_inj, "obs_time": obs_time, "nObs": nObs}

    return pedict, injdict, constants, param_names    

if __name__ == "__main__":
    ef = sys.argv[1]
    with open(ef, "rb") as pf:
        estimator = pickle.load(pf)

    label = ef[ef.index('r_')+2:ef.index('.pkl')]
    

    pedict, injdict, constants, param_names = load_pe_inj(sys.argv[2])
    T_obs = 1

    pe_prior = pedict['prior']
    (nevents, nsamples) = pe_prior.shape
    pe_prior = []
    print(param_names, nevents, nsamples)
    pe_samples = np.concatenate(tuple([pedict[n][:,:,None] for n in param_names[:-1]]), axis=-1).reshape(nevents*nsamples, len(param_names)-1)
    print(pe_samples.shape)
    pe_samples[:,0] = np.log(pe_samples[:,0])
    
    selection_samples = np.concatenate(tuple([injdict[n][:,None] for n in param_names[:-1]]), axis=-1)
    selection_samples[:, 0] = np.log(selection_samples[:,0])

    zmax = min([pe_samples[:,-1].max(), selection_samples[:,-1].max()])
    #zmin = max([pe_samples[:,-1].min(), selection_samples[:,-1].min()])
    samples = estimator.sample([1000000,]).cpu().detach().numpy()
    z_samples = samples[:,-1]
    log_dV_norm = np.log(Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples))+np.log(4*np.pi*T_obs)
    log_dV_norm = np.where(np.less_equal(z_samples, zmax), log_dV_norm, -np.inf)
    log_dV_norm = special.logsumexp(log_dV_norm) - np.log(len(samples))
    print(log_dV_norm)
    #sys.exit()
    inj_prior = injdict['prior']
    z_samples = pe_samples[:,-1]
    log_dV_pe = np.log(Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples))+np.log(4*np.pi*T_obs)
    log_dV_pe = np.where(np.less_equal(z_samples, zmax), log_dV_pe, -np.inf)
    z_samples = selection_samples[:,-1]
    log_dV_sel = np.log(Planck15.differential_comoving_volume(z_samples).to(u.Gpc**3/u.sr).value/(1+z_samples))+np.log(4*np.pi*T_obs)
    log_dV_sel = np.where(np.less_equal(z_samples, zmax), log_dV_sel, -np.inf)
    
    pm1q_pe = (batched_log_prob(estimator.bounded_log_prob,set_device(pe_samples), 10000).cpu().detach().numpy() + log_dV_pe - log_dV_norm - pe_samples[:,0]).reshape(nevents, nsamples)
    pm1q_inj = (estimator.bounded_log_prob(set_device(selection_samples)).cpu().detach().numpy()+log_dV_sel-log_dV_norm-selection_samples[:,0])
    #print(f'{label} log_surveyed_VT: {log_dV_norm}, log_vt: {special.logsumexp(pm1q_inj-np.log(inj_prior)) + log_dV_norm - np.log(constants["total_inj"])}')
    log_dV_norm_det = special.logsumexp(pm1q_inj-np.log(inj_prior)) + log_dV_norm - np.log(constants["total_inj"])
    samples = estimator.sample([100000,]).cpu().detach().numpy()
    with h5py.File(f'../__run__/weights_posydon_{label}.h5', 'w') as hf:
        hf.create_dataset('pe_logpdf', data = np.where(np.isnan(pm1q_pe), -np.inf, pm1q_pe))
        hf.create_dataset('inj_logpdf', data = np.where(np.isnan(pm1q_inj), -np.inf, pm1q_inj))
        hf.create_dataset('samples', data = samples)
        hf.create_dataset('vt', data = np.array([log_dV_norm]))
        hf.create_dataset('vt_det', data = np.array([log_dV_norm_det]))
    