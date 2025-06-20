"""
Hierarchical Mixture models for compact-binary rate and population models. Mixture of fixed PopSynth population and a flexible, inferable spline collector. This code is partially based on the spline-only model authored by Jaxen Godfrey <jaxen.godfrey@ligo.org>
"""
__author__= "Anarya Ray <anarya.ray@northwestern.edu>"

import numpyro
import numpyro
import numpyro.distributions as dist


from gwinferno.pipeline.analysis import hierarchical_likelihood
from gwinferno.pipeline.utils import bspline_mass_prior, bspline_redshift_prior

def m1qz_model(pedict, injdict, Nobs, Tobs, Ninj, mass_models, z_model, mmin, mmax, nspline_dict, param_names, rngkey=None, pm1qz_p_dict = None, popsynth_vt = None, horseshoe = False):
    """ Numpyro model

    Args:
        pedict (dict): dictionary of PE samples
        injdict (dict): dictionary of injection data
        Nobs (int): Number of CBC events
        Tobs (float): analysis time
        Ninj (int): total number of generated injectiosn
        mass_models (list of objs): list containing initialized b-splines for primary mass and mass ratio
        mag_model (obj): initialized b-spline for spin magnitude
        tilt_model (obj): initialized b-spline for cos_tilt
        z_model (obj): initialized b-spline-powerlaw for redshift
        mmin (float): minimum mass
        mmax (float): maximum mass
        nspline_dict (dict): dictionary containing the number of splines for each parameter
        param_names (list of str): list of parameters
        horseshoe (bool): whether or not to use horseshoe prior on spline coefficients
        pm1qz_p_dict (dict): dictionary containing popsynth weights for 'pe' and 'inj'
        popsynth_vt (float): surveyed spacetime volume within zmax for popsynth model

    """

    if pm1qz_p_dict == None:
        use_popsynth = False
        assert not horseshoe
    else:
        use_popsynth = True
        assert horseshoe
    
    #### Priors ####
    mass_cs, q_cs = bspline_mass_prior(
        m_nsplines = nspline_dict['m1'],
        q_nsplines = nspline_dict['q'],
        m_tau = 1,
        q_tau = 1,
        horseshoe = horseshoe,
    )

    if "redshift" in nspline_dict.keys():
        z_cs = bspline_redshift_prior(z_nsplines=nspline_dict['redshift'], z_tau = 1)
        z_spline = True
    else:
        z_spline = False
    
    lamb = numpyro.sample('lamb', dist.Normal(0,3))
    
    if use_popsynth:
        f = numpyro.sample("mixing frac", dist.Uniform(0., 1.))
    else:
        f = numpyro.deterministic("mixing frac", 0.)
    
    #### Calculate Weights ####
    

    def get_weights(datadict, pe_samples = True):
        

        p_m1q = mass_models(mass_cs, q_cs, pe_samples=pe_samples)
        

        p_z = z_model(datadict['redshift'], lamb) if not z_spline else z_model(datadict['redshift'], lamb, z_cs)
        if use_popsynth: # Construct Mixture model
            popsynth_key = "pe" if pe_samples else "injections"
            weights = (f * pm1qz_p_dict[popsynth_key] + (1.0-f) * p_m1q * p_z) / datadict['prior']
        else:
            weights = p_m1q * p_z / datadict['prior']

        return weights
        
    pe_weights = get_weights(pedict, pe_samples = True)
    inj_weights = get_weights(injdict, pe_samples = False)
    collector_v = numpyro.deterministic("collector_vt", z_model.normalization(lamb=lamb) if not z_spline else z_model.normalization(lamb=lamb, cs=z_cs))
    combined_v = numpyro.deterministic("combined_v", 1./(f/(popsynth_vt * 1e9 / Tobs) + (1.0-f)/collector_v)) if use_popsynth else collector_v

    #### Likelihood ####

    hierarchical_likelihood(
        pe_weights,
        inj_weights,
        float(Ninj),
        Nobs,
        Tobs,
        surveyed_hypervolume=combined_v,
        marginalize_selection = False,
        min_neff_cut = True,
        max_variance_cut = False,
        param_names = param_names,
        pedata = pedict,
        injdata = injdict,
        m2min = mmin,
        m1min = mmin,
        mmax = mmax,
        reconstruct_rate = True
    )