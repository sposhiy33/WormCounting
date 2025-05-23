CUDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM \
    --multiclass embryo \
    --num_classes 1 \
    --ce_coef 1 \
    --epochs 400 \
    --lr_drop 100 \
    --output_dir ./results/ \
    --lr 0.05 \
    --lr_backbone 0.00001 \
    --batch_size 3 \
    --num_patches 4 \
    --patch_size 512 \
    --eval_freq 1 \
    --gpu_id 0 \
    --row 1	--line 1 \
	--point_loss_coef 0.05 \
	--count_loss_coef 10.0 \
	--dense_loss_coef 1.0 \
	--distance_loss_coef 1.0 \
	--loss labels \
	--map_res 16 \
	--gauss_kernel_res 21 \
    --scale \
    --sharpness \
    --equalize \
    --gat
    # --resume results/multiclass_mixedimg/weights/best_mae.pth \
	#--freeze_regression
