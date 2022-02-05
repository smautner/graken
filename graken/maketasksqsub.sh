#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:30:00
#PBS -l mem=8gb
#PBS -S /bin/bash
#PBS -N Simple_Script_Job
#PBS -j oe
#PBS -o log/o_${PBS_JOBID}
#PBS -e log/e_${PBS_JOBID}
#PBS -q short
#PBS -V


# call qsub this -v iter=3 -t 0-99


source $HOME/.myconda/miniconda3/etc/profile.d/conda.sh
conda activate graken
cd $PBS_O_WORKDIR
mkdir -p tasks2_3
mkdir -p tasks2_4
mkdir -p tasks2_5
mkdir -p log
echo '###########'
pwd
echo '###########'


/home/fr/fr_fr/fr_sm1105/.myconda/miniconda3/envs/graken/bin/python maketasks/maketasks.py --ngraphs 30 --node_labels 4 --edge_labels 2\
	--bottleneck 2000 --iter $iter --numgr 3000  --graphsize 8 --out tasks2_$iter/task_${PBS_ARRAYID}

#/home/fr/fr_fr/fr_sm1105/.myconda/miniconda3/envs/graken/bin/python maketasks/maketasks.py --ngraphs 30 --node_labels 4 --edge_labels 2\
	#--bottleneck 200 --iter $iter --numgr 500  --graphsize 8 --out tasks2_${iter}/task_${PBS_ARRAYID}


