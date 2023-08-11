#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=graph_rnn
#SBATCH -t 2-00:00
#SBATCH --output=/home/steve.azzolin/GraphRNN_fork/sbatch_outputs/sbm_bigger.txt
#SBATCH --error=/home/steve.azzolin/GraphRNN_fork/sbatch_outputs/sbm_bigger.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`

eval "$(conda shell.bash hook)"
conda activate graphrnn


DATASET=sbm
EPOCH=2900

BIGGER=1
echo "Extracting with BIGGER=${BIGGER}"
python main.py --graph_type ${DATASET} --load True --load_epoch ${EPOCH} --bigger_graphs ${BIGGER}

BIGGER=2
echo "Extracting with BIGGER=${BIGGER}"
python main.py --graph_type ${DATASET} --load True --load_epoch ${EPOCH} --bigger_graphs ${BIGGER}

BIGGER=4
echo "Extracting with BIGGER=${BIGGER}"
python main.py --graph_type ${DATASET} --load True --load_epoch ${EPOCH} --bigger_graphs ${BIGGER}

BIGGER=8
echo "Extracting with BIGGER=${BIGGER}"
python main.py --graph_type ${DATASET} --load True --load_epoch ${EPOCH} --bigger_graphs ${BIGGER}




echo ${PATH}
echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes


# PLANAR
# incremento  1.5 NUMERO max nodi 96
# incremento  2.0 NUMERO max nodi 128
# incremento  4.0 NUMERO max nodi 256
# incremento  8.0 NUMERO max nodi 512