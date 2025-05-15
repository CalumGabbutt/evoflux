#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EVOFLUx is © 2025, Calum Gabbutt

EVOFLUx is published and distributed under the Academic Software License v1.0 (ASL).

EVOFLUx is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, email Calum Gabbutt at calum.gabbutt@icr.ac.uk. It is also published at https://github.com/gabor1/ASL/blob/main/ASL.md.

You may contact the original licensor at calum.gabbutt@icr.ac.uk.
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