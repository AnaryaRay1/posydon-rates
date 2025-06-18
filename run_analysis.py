"""
Inferring the mixture_models of ./hierarchical_models.py from GWTC data. This code is partially based on a spline-only inference script authored by Jaxen Godfrey <jaxen.godfrey@ligo.org>
"""

__author__= "Anarya Ray <anarya.ray@northwestern.edu>"
import os
os.environ["NPROC"]="4" 
os.environ["intra_op_parallelism_threads"]="1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from gwinferno.postprocess.calculations import calculate_bspline_mass_ppds, calculate_powerlaw_spline_rate_of_z_ppds, calculate_powerlaw_rate_of_z_ppds
from gwinferno.postprocess.plot import plot_mass_pdfs, plot_spin_pdfs, plot_rate_of_z_pdfs
from gwinferno.pipeline.utils import load_base_parser, posterior_dict_to_xarray, pdf_dict_to_xarray, load_pe_and_injections_as_dict, setup_bspline_mass_models, setup_powerlaw_spline_redshift_model
from gwinferno.models.parametric.parametric import PowerlawRedshiftModel


from models.hierarchical_models import m1qz_model as model

import arviz as az
import xarray as xr
import h5py

import jax.numpy as jnp
from jax.lax import fori_loop
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, DiscreteHMCGibbs


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import corner


def run_analysis(numpyro_model, pedict, injdict, constants, param_names, nspline_dict, parsargs, pm1qz_p_dict = None, skip_inference = False, IID = False, categorical = False, horseshoe = False, popsynth_vt = None):
    """run MCMC

    Args:
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injecitons
        constants (dict): dictionary of relevant constants
        param_names (list of strs): list of parameters
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        parsargs (ArgumentParser): args from ArgumentParser.parse_args()
        skip_inference (bool, optional): If True, does not perform inference. Defaults to False.

    Returns:
        if skip_inference == False:
            posterior (dict): dictionary of posterior samples
            z_model (obj): redshift model (needed for later calculations)
        if skip_inference == True:
            z_model
    """
    if "redshift" in nspline_dict.keys():
        z_model = setup_powerlaw_spline_redshift_model(pedict, injdict, nspline_dict['redshift'])
    else:
        z_model = PowerlawRedshiftModel(pedict["redshift"], injdict["redshift"])
        
    mass_models = setup_bspline_mass_models(pedict, injdict, nspline_dict['m1'], nspline_dict['q'], mmin = parsargs.mmin, mmax = parsargs.mmax)
    
    if not skip_inference:
        nChains = parsargs.chains
        numpyro.set_host_device_count(nChains)

        if categorical:
            innerkernel = NUTS(numpyro_model)
            kernel = DiscreteHMCGibbs(innerkernel)
            mcmc = MCMC(kernel, num_warmup=parsargs.warmup, num_samples=parsargs.samples, num_chains=nChains)
        else:
            kernel = NUTS(numpyro_model)
            mcmc = MCMC(kernel, num_warmup=parsargs.warmup, num_samples=parsargs.samples, num_chains=nChains)

        rng_key = random.PRNGKey(parsargs.rngkey)
        rng_key, catkey, rng_key_ = random.split(rng_key, num=3)

        mcmc.run(rng_key_, pedict, injdict, constants['nObs'], constants['obs_time'], constants['total_inj'], mass_models, z_model, parsargs.mmin, parsargs.mmax, nspline_dict, param_names, rngkey=catkey, pm1qz_p_dict = pm1qz_p_dict, horseshoe = horseshoe, popsynth_vt = popsynth_vt)
        posterior = mcmc.get_samples()

        return posterior, z_model

    else:
        return z_model

def setup_result_dir(parsargs):
    """construct a directory to save results to

    Args:
        parsargs (): args from argument parser

    Returns:
        label (str): label for file names
        full_dir (str): result directory
    """
    label = parsargs.run_label + f'_{parsargs.warmup}w_{parsargs.samples}s_rng{parsargs.rngkey}'
    result_directory = '__run__/' + parsargs.result_dir+ '/' + parsargs.run_label
    full_dir = f'{result_directory}/rngnum-{parsargs.rngkey}/{parsargs.warmup}w_{parsargs.samples}s'
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    print(f'result files will be saved in directory: {full_dir}')
    return label, full_dir

def main():

    """
    load argument parser (used when running script from command line)
    """

    base_parser = load_base_parser()

    ### example of function that adds additional arguments to the base parser.
    def add_args(parser):
        parser.add_argument('--popsynth-file', type=str, default = '')
        parser.add_argument('--horseshoe', type=str, default = 'False')
        parser.add_argument('--zspline', type=str, default = 'False')
        return parser

    parser = add_args(base_parser)
    args = parser.parse_args()
    horseshoe = args.horseshoe == "True"
    zspline = args.zspline == "True"
    popsynth_file = args.popsynth_file
        
    if popsynth_file == '' :
        use_popsynth = False
        popsynth_samples = None    
        pm1qz_p_dict = None
        popsynth_vt = None
    else:
        use_popsynth = True
        pm1qz_p_dict = {}
        with h5py.File(popsynth_file, "r") as hf:
            pm1qz_p_dict["pe"] = jnp.array(hf["pe_logpdf"])
            pm1qz_p_dict["injections"] = jnp.array(hf["inj_logpdf"])
            pm1qz_p_dict = {k:jnp.where(jnp.isnan(lpdf)*jnp.isinf(lpdf), 0., jnp.exp(lpdf)) for k, lpdf in pm1qz_p_dict.items()}
            popsynth_samples = hf['samples'][()]
            popsynth_samples[:,0] = np.exp(popsynth_samples[:,0])
            popsynth_vt = np.exp(hf['vt'][()])[0]

    if zspline:
        nspline_dict = {
        'm1': args.m_nsplines,
        'q': args.q_nsplines,
        'redshift': args.z_nsplines,
        }
    else:
        nspline_dict = {
        'm1': args.m_nsplines,
        'q': args.q_nsplines,
        }
        

    """
    Load PE and injections as dictionaries, along constants like # of observations, 
    injection analysis time, etc., and a list of the parameter names being modeled.
    """

    pedict, injdict, constants, param_names = load_pe_and_injections_as_dict(args.pe_inj_file)
    

    """
    Setup directory where results will be stored.
    """
    label, result_dir = setup_result_dir(args)
    label = label + ("_horseshoe" if horseshoe else '') + ("_zspline" if zspline else '')

    """
    Run inference and save posterior samples to file. If flag --skip-inference present, then don't perform inference and load posterior samples from existing file.
    """

    if args.skip_inference:
        z_model = run_analysis(model, pedict, injdict, constants, param_names, nspline_dict, args, skip_inference = True, horseshoe = horseshoe, popsynth_vt = popsynth_vt)
        print(f'loading posterior file: {result_dir}/{label}_posterior_samples.h5')
        posterior = xr.load_dataset(result_dir + f"/{label}_posterior_samples.h5")

    else:
        posterior_dict, z_model = run_analysis(model, pedict, injdict, constants, param_names, nspline_dict, args, pm1qz_p_dict = pm1qz_p_dict, horseshoe = horseshoe, popsynth_vt = popsynth_vt)
        print(f'posteriors file saved: {result_dir}/{label}_posterior_samples.h5')
        posterior = posterior_dict_to_xarray(posterior_dict)
        posterior.to_netcdf(result_dir + f"/{label}_posterior_samples.h5", engine='scipy')
    

    
    
    """
    Save trace plot
    """

    '''
    idata = az.InferenceData(posterior=posterior.expand_dims(dim={'chain': 1}))
    az.rcParams['plot.max_subplots'] = 200
    az.plot_trace(idata)
    plt.tight_layout()
    plt.show()
    plt.savefig(result_dir + f"/trace_plot_{label}.png", dpi=100)
    '''

    """
    Create list of population labels and corresponding colors. In this analysis, we are fitting the entire population with one model, so we only have 1 element. 
    
        Example of model with 2 Subpopulations: 
            names = ['Population A', 'Population B']
            colors = ['red', 'blue']
    """
    names = ['B-Spline']
    colors = ['tab:blue']

    print('plotting rates:')
    R = posterior['rate'].values
    Ndet = posterior["unscaled_rate"].values
    Ntot = Ndet/posterior["detection_efficiency"].values

    if use_popsynth:
        f_prior = np.random.uniform(0., 1., args.samples * args.chains)
        f = posterior['mixing frac'].values
        np.savetxt(f"{result_dir}/f.txt", np.c_[f,f_prior])
        combined_vt = posterior['surveyed_hypervolume'].values
        collector_vt = posterior['collector_vt'].values / 1e9
        plt.clf()
        _ = plt.hist(np.log(collector_vt * constants["obs_time"]), label = 'collector', histtype = 'step', density = True) 
        _ = plt.hist(np.log(combined_vt), label = 'combined', histtype = 'step', density = True)
        plt.axvline(np.log(popsynth_vt), color = "green", label = "popsynth")
        plt.legend()
        
        plt.savefig(f"{result_dir}/vts.png")
        plt.clf()
        fig = corner.corner(np.array([combined_vt, posterior["lamb"].values]).T)
        fig.savefig(f"{result_dir}/vtc_lamb.png")
        plt.clf()
        #Rp = R*f*combined_vt/popsynth_vt#/constants["obs_time"]
        Rp = Ntot * f /popsynth_vt
         
        Rc = Ntot * (1.0-f) / collector_vt  / constants["obs_time"] 
        R = Rc + Rp
        _ = plt.hist(f, histtype = 'step', density = True, label = "posterior" )
        _ = plt.hist(f_prior, histtype = 'step', density = True, label = "prior" )
        plt.legend()
        plt.savefig(f"{result_dir}/fraction.png")
        fig = corner.corner(np.array([Rp,Rc]).T, histkwargs = {"density" : True}, bins = 15, labels = [r"$R_{p}$", r"$R_{c}$"]) 
        fig.savefig(f"{result_dir}/rates.png")
    else:
        Rc = R
        Rp = 0
    plt.clf()
    _ = plt.hist(jnp.log(Ntot), histtype = 'step', density = True, label = "Total" )
    _ = plt.hist(jnp.log(Ndet), histtype = 'step', density = True, label = "Observed" )
    plt.legend()
    plt.savefig(f"{result_dir}/log_counts.png")
    plt.clf()
    plt.hist(R, histtype = 'step', density = True)
    plt.savefig(f"{result_dir}/rate_combined.png")

    """
    Calculate Mass pdfs (for loop necessary for multiple subpopulations)
    """

    print('calculating mass ppds:')
    mass_pdfs, m1s, q_pdfs, qs = calculate_bspline_mass_ppds(posterior[f'mass_cs'].values, posterior[f'q_cs'].values, nspline_dict, 5.5, args.mmax, pop_frac = 1.0 - posterior['mixing frac'].values if use_popsynth else None)
    
    mass_pdfs = [mass_pdfs]
    q_pdfs = [q_pdfs]

    """
    Calculate rate as a funciton of redshift
    """
    print('calculating rate(z) ppds:')
    if not zspline:
        r_of_z, zs = calculate_powerlaw_rate_of_z_ppds(posterior['lamb'].values, Rc, z_model)
    else:
        r_of_z, zs = calculate_powerlaw_spline_rate_of_z_ppds(posterior['lamb'].values, posterior['z_cs'].values, Rc, z_model) 
    

    """
    Save PDF plots of each parameter
    """
    print('plotting mass distributions:')
    plot_mass_pdfs(mass_pdfs, q_pdfs, m1s, qs, names, label, result_dir, save = args.save_plots, colors = colors, popsynth_frac = posterior["mixing frac"].values if use_popsynth else None, popsynth_samples = popsynth_samples[:,:2] if use_popsynth else None)
    

    print('plotting redshift distributions:')
    plot_rate_of_z_pdfs(r_of_z, zs, label, result_dir, save = args.save_plots, popsynth_frac = Rp if use_popsynth else None, popsynth_samples = popsynth_samples[:,-1] if use_popsynth else None)

if __name__ == '__main__':
    main()
