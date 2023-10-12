#!/bin/sh

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