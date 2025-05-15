#!/usr/bin/env python3

"""
EVOFLUx is © 2025, Calum Gabbutt

EVOFLUx is published and distributed under the Academic Software License v1.0 (ASL).

EVOFLUx is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

You should have received a copy of the ASL along with this program; if not, email Calum Gabbutt at calum.gabbutt@icr.ac.uk. It is also published at https://github.com/gabor1/ASL/blob/main/ASL.md.

You may contact the original licensor at calum.gabbutt@icr.ac.uk.
"""

import pandas as pd
import evoflux.evoloo as el
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Calculate leave-one-out')   
    parser.add_argument('--paths', nargs='+', required=True, help='File paths')
    parser.add_argument('--labels', nargs='+', default = None)                                          
    parser.add_argument('--outputdir', type=str, default='', 
                        help='path to folder in which to store output')
    parser.add_argument('--sample', type=str, default='example',
                        help='samplename') 

    # Execute the parse_args() method
    args = parser.parse_args()

    inference_paths = args.paths
    labels = args.labels
    outputdir = args.outputdir
    sample = args.sample

    inference_list = [el.load_inference(p) for p in inference_paths]

    model_compare = el.model_selection(inference_list, outputdir, sample, labels)

if __name__ == "__main__":
    main()