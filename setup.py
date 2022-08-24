#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import glob

scripts = glob.glob("scripts/*.py")

setup(
    name="flipflopblood",
    url="https://github.com/CalumGabbutt/flipflopblood",
    version=1.0,
    author="Calum Gabbutt",
    author_email="calum.gabbutt@icr.ac.uk",
    packages=["flipflopblood"],
    license="MIT",
    scripts=scripts,
    description=("A Bayesian pipeline to infer stem cell"
                 "dynamics from methylation array data."),
    install_requires=["numpy", "scipy", "matplotlib", "pandas", 
                    "dynesty", "joblib", "seaborn", "arviz"],
    package_data={
        "flipflopblood": ["files/*"],
    }
)