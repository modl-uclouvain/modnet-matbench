mkdir -p final_plots
for f in $(ls | grep "matbench_"); do
    echo $f
    cp $f/plots/*.pdf final_plots/
done

