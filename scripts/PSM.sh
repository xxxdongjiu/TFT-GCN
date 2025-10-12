python -u run.py  --is_training 1 \
    --root_path /home/xusheng/dataset/PSM \
    --data PSM \
    --seq_len 100 \
    --enc_in 25 \
    --c_out 25 \
    --d_model 64 \
    --anomaly_ratio 1.5 \
    --batch_size 256 \
    --train_epochs 5