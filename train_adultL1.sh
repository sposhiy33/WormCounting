CUDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM \
	--multiclass L1 \
	--num_classes 1 \
	--ce_coef 1 \
	--epochs 150 \
    --lr_drop 150 \
    --output_dir ./results/ \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 4 \
    --eval_freq 1 \
    --gpu_id 0 \
    --row 1	--line 1 \
	--label_loss_coef 1.0 \
	--point_loss_coef 0.01 \
	--dense_loss_coef 1.0 \
	--distance_loss_coef 1.0 \
	--loss labels points \
	--map_res 16 \
	--gauss_kernel_res 21 \
	# --resume results/multiclass_mixedimg/weights/best_mae.pth \
	# --freeze_regression
