# posydon-rates
To run inference, clone [gwinferno](https://github.com/FarrOutLab/GWInferno) and apply patch. Then setup an environment and install gwinferno according to the instructions in gwinferno's [readme](https://github.com/FarrOutLab/GWInferno?tab=readme-ov-file#for-gpu). Note you will skip the cloning part since that is already done. Once installed, to run analysis:


#### Collector (spline-only):
```
python fetch_data.py # Works on CIT cluster.
python run_analysis.py --pe-inj-file ../data/pe-inj-data-o3_m1qz.h5 --run-label m1qz_mmin3_collector_only --mmin 3 --mmax 100 --samples 4000 --warmup 4000 --chains 4 --rngkey 129 --save-plots True --result-dir __run__/results_m1qz 
```

### Mixture-model
```
python run_analysis.py --pe-inj-file ../data/pe-inj-data-o3_m1qz.h5 --run-label m1qz_mmin3_with_popsynth --mmin 3 --mmax 100 --samples 4000 --warmup 4000 --chains 4 --rngkey 129 --save-plots True --result-dir __run__/results_m1qz --popsynth-file /path/to/popsynth/weights/ --horseshoe=True 
```

Models are coded up in ```models/hierarchical_models.py```

More detailed instructions coming soon.


