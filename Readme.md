# pytorch-normalizing-flows

Implementions of normalizing flows (NICE, RealNVP, MAF, IAF, Neural Splines Flows, etc) in PyTorch. The original codes are modified for the term project of MIT 6.862 spring 2021 .

**todos**
- TODO: 2D -> 3D
- TODO: 3D -> ND
- TODO: Use other flow models (multi-scale architectures, Glow nets, IAF, etc)


## To launch in MIT EAPS cluster (eofe7.mit.edu)

To setup the environment for the first time, execute followings.
```
ssh eofe7.mit.edu

cd /nobackup1c/users/$USER/
git clone https://github.com/6862-2021SP-team3/pytorch-normalizing-flows
cd pytorch-normalizing-flows
wget -O pi0.pkl https://www.dropbox.com/s/hrdhr5o1khtclmy/pi0.pkl?dl=0

module load python/3.6.3
pip install --user numpy, pandas, matplotlib, pickle5, torch
module load engaging/OpenBLAS/0.2.14
module load engaging/torch/20160128
```

From the second time, use
```
ssh eofe7.mit.edu
cd /nobackup1c/users/$USER/pytorch-normalizing-flows
module load python/3.6.3
module load engaging/OpenBLAS/0.2.14
module load engaging/torch/20160128
python clas12-nflow.py
```

This will be updated with the `srun` and `sbatch` script (ongoing).
From admin's comment, cuda is only available with slurm commands and centos6.