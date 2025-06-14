CUDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM \
    --multiclass embryo \
    --num_classes 1 \
    --ce_coef 1 \
    --epochs 200 \
    --lr_drop 100 \
    --output_dir ./results/ \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 3 \
    --num_patches 4 \
    --patch_size 512 \
    --eval_freq 1 \
    --gpu_id 1 \
    --row 1 \
    --line 1 \
    --point_loss_coef 0.0002 \
    --aux_loss_coef 0.2 \
    --loss labels points \
    --scale \
    --sharpness \
    --equalize \
    --mlp
    # --resume results/multiclass_mixedimg/weights/best_mae.pth \
	#--freeze_regression
