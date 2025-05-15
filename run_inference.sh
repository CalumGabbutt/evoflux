#!/bin/sh


# EVOFLUx is © 2025, Calum Gabbutt

# EVOFLUx is published and distributed under the Academic Software License v1.0 (ASL).

# EVOFLUx is distributed in the hope that it will be useful for non-commercial academic research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.

# You should have received a copy of the ASL along with this program; if not, email Calum Gabbutt at calum.gabbutt@icr.ac.uk. It is also published at https://github.com/gabor1/ASL/blob/main/ASL.md.

# You may contact the original licensor at calum.gabbutt@icr.ac.uk.


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