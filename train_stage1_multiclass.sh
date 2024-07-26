CUDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM --multiclass --num_classes 2 --ce_coef 0.7 1 \
    --epochs 250 \
    --lr_drop 250 \
    --output_dir ./results/ \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 4 \
    --eval_freq 1 \
    --gpu_id 0 
