declare -a point=("0.01" "0.001" "0.0001")
declare -a eos=("1.0" "0.75" "0.5" "0.25" "0.0")
declare -a coef=("1.0" "10.0" "100.0")
declare -a map=("4" "8" "16" "32" "64" "128")

# for L1 samples
for i in "${coef[@]}"
do
	for j in "${map[@]}"
	do
		echo "{$i}_{$j}_1"
		python total_test.py --weight_path results/DENSE_search_{$i}_{$j}_1/weights/best_training_loss.pth --multiclass --num_classes 1 --class_filter 1 --row 1 --line 1
	done
done
