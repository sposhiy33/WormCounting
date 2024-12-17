declare -a point=("0.01" "0.001" "0.0001")
declare -a eos=("1.0" "0.75" "0.5" "0.25" "0.0")
declare -a map=("4" "8" "16" "32" "64" "128")
declare -a coef=("1.0" "10.0" "100.0")

# for L1 samples
for i in "${coef[@]}"
do
	for j in "${map[@]}"
	do
		bash train_stage1_multiclass.sh DENSE_search_{$i}_{$j}_1 dataroot/resize_mixed_L1 $i $j
		echo "{$i}_{$j}_1"
	done
done
