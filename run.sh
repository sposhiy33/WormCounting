
CUDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM --num_classes 2 --multiclass --ce_coef 0.5 1\
    --epochs 1000 \
    --lr_drop 1000 \
    --output_dir ./results/ \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 1 \
    --eval_freq 1 \
    --gpu_id 0  
