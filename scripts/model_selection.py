#!/usr/bin/env python3

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

    el.model_selection(inference_list, outputdir, sample, labels)

if __name__ == "__main__":
    main()