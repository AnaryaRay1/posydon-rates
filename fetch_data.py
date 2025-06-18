__author__= "Jaxen Godfrey <jaxen.godfrey@ligo.org>"


import os
os.environ["NPROC"]="4" 
os.environ["intra_op_parallelism_threads"]="1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" 
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="false" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


import h5py
import json
from glob import glob

from gwinferno.preprocess.data_collection import (
    load_injection_dataset,
    load_posterior_dataset,
    save_posterior_samples_and_injection_datasets_as_idata,
)


def collect():
    with open(
        '/home/jaxen.godfrey/FarrOutLab/GWInferno/work/gwinferno_review/'
        'data_collection/bbh_gwtc3.json',
        'r',
    ) as f:
        json_dict = json.load(f)
    
    run_map = {}
    ev_id = json_dict['events'].keys()
    
    for ev in ev_id:
        catalog = json_dict['events'][ev]['catalog.shortName']
        common_name = json_dict['events'][ev]['commonName']
    
        if catalog == 'GWTC-1-confident':
            file_path = (
                '/home/rp.o4/catalogs/GWTC-1/GWTC-1_sample_release/'
                f'{common_name}_GWTC-1.hdf5'
            )
            waveform = 'Overall_posterior'
            redshift_prior = 'euclidean'
            catalog_name = 'GWTC-1'
    
        if catalog == 'GWTC-2.1-confident':
            name = ev.split('-v')[0]
            file_path = (
                '/home/rp.o4/catalogs/GWTC-2.1/data-release/'
                f'IGWN-GWTC2p1-v2-{name}_PEDataRelease_mixed_cosmo.h5'
            )
            waveform = 'C01:Mixed'
            redshift_prior = 'comoving'
            catalog_name = 'GWTC-2.1'
    
        if catalog == 'GWTC-3-confident':
            name = ev.split('-v')[0]
            file_path = (
                '/home/rp.o4/catalogs/GWTC-3/data-release/'
                f'IGWN-GWTC3p0-v1-{name}_PEDataRelease_mixed_cosmo.h5'
            )
            waveform = 'C01:Mixed'
            redshift_prior = 'comoving'
            catalog_name = 'GWTC-3'
    
        run_map[common_name] = {
            'file_path': file_path,
            'waveform': waveform,
            'redshift_prior': redshift_prior,
            'catalog': catalog_name,
        }

    with open('./data/full_catalog_keys_to_read-o3.json', 'w') as out_file:
        json.dump(run_map, out_file)


def save():
    inj_file = (
        '/home/anarya.ray/gppop-mdc/3d-clean/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5'
    )
    pe_key_file = './data/full_catalog_keys_to_read-o3.json'
    file_name = './data/pe-inj-data-o3_m1qz.h5'
    param_names = [
        'mass_1', 'mass_ratio', 'redshift',
    ]

    inj_data = load_injection_dataset(
        inj_file, param_names, through_o4a = False, through_o3 = True,
    )
    max_mass = inj_data.injections.sel(param='mass_1').max().values
    pe_data = load_posterior_dataset(
        maximum_mass = max_mass,
        key_file = pe_key_file,
        param_names = param_names,
    )
    
    save_posterior_samples_and_injection_datasets_as_idata(
        file_name, pe_data, inj_data,
    )


def main():
    collect()
    save()


if __name__ == '__main__':
    main()
