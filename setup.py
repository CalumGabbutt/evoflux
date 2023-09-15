#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import glob

scripts = glob.glob("scripts/*.py")

setup(
    name="evoflux",
    url="https://github.com/CalumGabbutt/evoflux",
    version=1.0,
    author="Calum Gabbutt",
    author_email="calum.gabbutt@icr.ac.uk",
    packages=["evoflux"],
    license="MIT", # <- change this!
    scripts=scripts,
    description=("A Bayesian pipeline to infer a cancer's evolutionary history"
                 "dynamics from methylation array data."),
    install_requires=["numpy", "scipy", "matplotlib", "pandas", 
                    "dynesty", "joblib", "seaborn", "arviz"],
    package_data={
        "evoflux": ["files/*"],
    }
)