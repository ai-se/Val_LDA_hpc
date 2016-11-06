
#! /bin/tcsh



for VAR in academia apple anime android scifi SE0 SE1 SE2 SE3 SE4
do
  for MES in LDAl2_SMOTE_100 LDA_SMOTE_100 LDAl2_SMOTE_200 LDA_SMOTE_200
  do
    bsub -q standard -W 2400 -n 5 -o ./out/$VAR$MES.out.%J -e ./err/$VAR$MES.err.%J mpiexec -n 5 /share2/zyu9/miniconda/bin/python2.7 val_lda.py exp_hpc $MES $VAR
  done
done
