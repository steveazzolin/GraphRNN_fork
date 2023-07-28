#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=graph_rnn
#SBATCH -t 1:00:00
#SBATCH --output=/home/steve.azzolin/GraphRNN_fork/sbatch_outputs/planar.txt
#SBATCH --error=/home/steve.azzolin/GraphRNN_fork/sbatch_outputs/planar.txt
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


DATASET=sbm
BIGGER=200
python main.py --graph_type planar


echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes