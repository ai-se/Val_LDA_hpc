rm ./err/*
rm ./out/*

for VAR in academia apple anime android scifi SE0 SE1 SE2 SE3 SE4
do
  bsub -q standard -W 2400 -n 10 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J mpiexec -n 10 /share2/zyu9/miniconda/bin/python2.7 tune_LDA.py exp $VAR
done
