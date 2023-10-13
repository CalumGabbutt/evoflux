# EVOFLUx

# Installation 
`evoflux` is compatible with Python 3.9 It requires `numpy` (basic maths functions), `scipy` (special maths functions), `pandas` (dataframes manipulation), `dynesty` (nested sampling - Bayesian inference), `joblib` (pickling large data files, preserving dynesty structure), `notebook` (Jupyter notebook for ), `matplotlib` (plotting), `seaborn` (plotting), `arviz` and `corner`  (Bayesian plotting)

The package can be installed directly from a local copy of the Github repo. We reccommend installing evolfux in a virtual environment, using venv, and pip to install the dependencies (conda can also be used):

```
git clone https://github.com/CalumGabbutt/evoflux.git
cd evoflux
python3 -m venv evolfuxenv
source evolfuxenv/bin/activate
pip install -r requirements.txt
python3 setup.py install
```

# Usage
The functions used to perform the Bayesian inference can be imported in python using the command
`from evolfux import evolfux as ev`. Simialrly, the plotting functions can be imported as `import evoflux.evoplots as ep` and the leave-one-out model selection functions can be imported with `import evoflux.evoloo as el`.

An example script `run_inference.sh` has been included, which performs the inference upon the example data in `data/`.  To run the example script, run the following code snippet from the command line, whilst located in the main directory copied from the Gitgub repo:
`./run_inference.sh`

(if a permission denied error is returned, run the command `chmod +x run_inference.sh` to grant access to the script)

An example Jupyter notebook is included under `evoflux_notebook.ipynb`. 