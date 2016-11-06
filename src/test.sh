

for MES in SVM LDA LDAl2
do
  bsub -q standard -W 2400 -n 5 -o ./out/SE0$MES.out.%J -e ./err/SE0$MES.err.%J mpiexec -n 5 /share2/zyu9/miniconda/bin/python2.7 val_lda.py exp_hpc $MES SE0
done

