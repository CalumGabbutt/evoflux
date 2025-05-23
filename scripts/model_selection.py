#!/usr/bin/env python3

"""
Copyright 2025 The Institute of Cancer Research.

Licensed under a software academic use license provided with this software package (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: https://github.com/CalumGabbutt/evoflux
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
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