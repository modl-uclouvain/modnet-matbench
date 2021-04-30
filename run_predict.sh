tasks=(
    steels
    elastic
    expt_gap
    dielectric
    jdft2d
    phonons
)
for t in "${tasks[@]}"; do
    echo $t
    python run_benchmark.py --task $t --predict --plot
done
