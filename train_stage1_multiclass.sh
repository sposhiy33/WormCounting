UDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM --multiclass --num_classes 1 --ce_coef 1 \
    --epochs 150 \
    --lr_drop 150 \
    --output_dir ./results/ \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 1 \
    --eval_freq 1 \
    --gpu_id 0 \
    --row 1	--line 1 \
	--point_loss_coef 0.0002 \
	--class_filter 1
	# --resume results/multiclass_mixedimg/weights/best_mae.pth \
	# --freeze_regression
