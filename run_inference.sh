#!/bin/sh

datafile=data/CLL_beta_noob_subset_fcpgs.csv
patientinfofile=data/BloodMethMetadata.csv
outputdir=examples
samplename=CLL.1
nlive=300
thetamean=3
thetastd=2

mkdir -p ${outputdir}

inference.py $datafile $patientinfofile $outputdir $samplename --verbose -nlive $nlive -thetamean $thetamean -thetastd $thetastd

plot_posterior.py $datafile $patientinfofile $outputdir $samplename -thetamean $thetamean -thetastd $thetastd