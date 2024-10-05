#!/bin/bash -l
base_job_name="nasal"
job_file="job.sh"
expnm="diffal"
a_count=0
acqs=(r e c b g m p)
# acqs=(g)
# dats=(m s)
dats=(fm ct)
netw="mlp"
aug=0
declare -a ac_arr=("random" "entropy" "coreset" "bald" "badge" "margin" "least_confidence")
# declare -a ac_arr=("badge")
# declare -a dt_arr=("MNIST" "FashionMNIST" "CIFAR10" "SVHN")
declare -a dt_arr=("FashionMNIST" "CIFAR10")
for tmp_acq in "${ac_arr[@]}"
do
    d_count=0
    for tmp_data in "${dt_arr[@]}"
    do
        job_name="nas-${dats[$d_count]}-${acqs[$t_count]}-i1000_rep"
        out_file="nas-${dats[$d_count]}-${acqs[$t_count]}-i1000_rep.out"
        err_file="nas-${dats[$d_count]}-${acqs[$t_count]}-i1000_rep.err"
        job_name="nas-${dats[$d_count]}-${acqs[$t_count]}-i1000_rep"

        # echo "sn-${acqs[$t_count]}-${dats[$d_count]}, $tmp_data, $tmp_acq"
    
        export dataset=$tmp_data
        export acq_nas="$tmp_acq-nas-$tmp_data"
        export acq=$tmp_acq
        export $netw
        # echo "$dataset-$acq"
        sbatch -J $job_name -o $out_file -t 04-12:00:00 -p tier3 -e $err_file $job_file
        ((d_count=d_count+1))
    done
    ((t_count=t_count+1))
done

