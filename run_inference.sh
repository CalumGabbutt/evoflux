#!/bin/sh


# Copyright 2025 The Institute of Cancer Research.
#
# Licensed under a software academic use license provided with this software package (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at: https://github.com/CalumGabbutt/evoflux
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


datafile=data/beta_fcpgs.csv
patientinfofile=data/BloodMethMetadata.csv
mode=neutral
outputdir=examples/$mode
samplename=SCLL-059
nlive=100
NSIM=2000
sample_meth=rwalk
mkdir -p ${outputdir}

inference.py $datafile $patientinfofile $outputdir $samplename --verbose -nlive $nlive -NSIM $NSIM -sample_meth $sample_meth -mode $mode

plot_posterior.py $datafile $patientinfofile $outputdir $samplename -NSIM $NSIM -mode $mode
calculate_loo.py $datafile $patientinfofile $outputdir $samplename -mode $mode