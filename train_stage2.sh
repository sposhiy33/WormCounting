CUDA_VISIBLE_DEVICES=0 python train_classifer.py --expname $1 --data_root $2 \
    --dataset_file WORM --multiclass --ce_coef 0.7 1 --point_weights $3\
    --epochs 250  --downstream_num_classes 3\
    --lr_drop 250 \
    --output_dir ./results/ \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0  