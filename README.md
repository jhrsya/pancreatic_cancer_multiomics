## 1. Setup the environment

### 1) Download the repository
```
git clone git@github.com:jhrsya/pancreatic_cancer_multiomics.git
cd pancreatic_cancer_multiomics
```

### 2) Create a conda environment
```
conda create -n multiomics python=3.9.13
conda activate multiomics
pip install -r requirements.txt
```

## 2. Reproduce experiment results
```
python co_112.py
python co_211.py
```


