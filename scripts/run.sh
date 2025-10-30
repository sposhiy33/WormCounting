CUDA_VISIBLE_DEVICES=1 python src/training/train.py --expname $1 --data_root $2 \
    --dataset_file WORM \
    --multiclass embryo Gravid \
    --num_classes 2 \
    --ce_coef 1.0 1.0 \
    --epochs 200 \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 3 \
    --num_patches 4 \
    --patch_size 512 \
    --eval_freq 1 \
    --gpu_id 0 \
    --row 1 \
    --line 1 \
    --point_loss_coef 0.0002 \
    --aux_loss_coef 0.2 \
    --loss labels \
    --scale \
    --sharpness \
    --equalize \
    --architecture mlp_classifier \
    # --resume results/multiclass_mixedimg/weights/best_mae.pth \
	# --freeze_regression \
    # --salt_and_pepper \
