#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=graphrnn
#SBATCH -t 2-00:00
#SBATCH --output=/home/steve.azzolin/GraphRNN_fork/sbatch_outputs/comm_nopad.txt
#SBATCH --error=/home/steve.azzolin/GraphRNN_fork/sbatch_outputs/comm_nopad.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`

set -e
export PATH="/nfs/data_chaos/sazzolin/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate graphrnn


python main.py --graph_type comm


echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes