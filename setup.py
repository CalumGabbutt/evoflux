#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright 2025 The Institute of Cancer Research.

Licensed under a software academic use license provided with this software package (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: https://github.com/CalumGabbutt/evoflux
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

from setuptools import setup
import glob

scripts = glob.glob("scripts/*.py")

setup(
    name="evoflux",
    url="https://github.com/CalumGabbutt/evoflux",
    version=1.1,
    author="Calum Gabbutt",
    author_email="calum.gabbutt@icr.ac.uk",
    packages=["evoflux"],
    license="ASL",
    scripts=scripts,
    description=("A Bayesian pipeline to infer a cancer's evolutionary history"
                 "dynamics from methylation array data."),
    install_requires=["numpy", "scipy", "matplotlib", "pandas", 
                    "dynesty", "joblib", "seaborn", "arviz"],
    package_data={
        "evoflux": ["files/*"],
    }
)