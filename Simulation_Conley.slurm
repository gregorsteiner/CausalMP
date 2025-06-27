#!/bin/bash
#SBATCH --nodes=1 # DON'T CHANGE THIS!!!
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3999
#SBATCH --time=48:00:00
 
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
#SBATCH --mail-user=gregor.steiner@warwick.ac.uk
 
module purge
module load GCC/11.3.0 OpenMPI/4.1.4 R/4.2.1
 
# Set number of cores used by parallel package
export MC_CORES=$SLURM_CPUS_PER_TASK
 
# Execute the R script
Rscript Simulation_Conley.R