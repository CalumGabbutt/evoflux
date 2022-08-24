# flipflopblood

# Installation 
`flipflopblood` is compatible with Python 3.9 It requires `numpy` (basic maths functions), `scipy` (special maths functions), `pandas` (dataframes manipulation), `dynesty` (nested sampling - Bayesian inference), `joblib` (pickling large data files, preserving dynesty structure),`matplotlib` (plotting), `seaborn` (plotting), `arviz` and `corner`  (Bayesian plotting)

The package can be installed directly from a local copy of the Github repo. We reccommend installing flipflopblood in a virtual environment, using venv, and pip to install the dependencies (conda can also be used):

```
git clone https://github.com/CalumGabbutt/flipflopblood.git
cd flipflopblood
python3 -m venv flipflopbloodenv
source flipflopbloodenv/bin/activate
pip install -r requirements.txt
python3 setup.py install
```

# Usage
The functions used to perform the Bayesian inference can be imported in python using the command
`from flipflopblood import flipflopblood as fb`

An example script `run_inference.sh` has been included, which performs the inference upon the example data in `data/`.  To run the example script, run the following code snippet from the command line, whilst located in the main directory copied from the Gitgub repo:
`./run_inference.sh`

(if a permission denied error is returned, run the command `chmod +x run_inference.sh` to grant access to the script)