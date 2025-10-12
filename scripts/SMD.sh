python -u run.py  --is_training 1 \
    --root_path /home/xusheng/dataset/SMD \
    --data SMD \
    --seq_len 100 \
    --enc_in 38 \
    --c_out 38 \
    --d_model 64 \
    --anomaly_ratio 0.5 \
    --batch_size 128 \
    --train_epochs 5