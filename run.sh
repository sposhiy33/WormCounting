CUDA_VISIBLE_DEVICES=0 python train.py --expname $1 --data_root $2 \
    --dataset_file WORM \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./results/ \
	--lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
