python -u run.py  --is_training 1 \
    --root_path /home/xusheng/dataset/SKAB \
    --data SKAB \
    --seq_len 100 \
    --enc_in 8 \
    --c_out 8 \
    --d_model 64 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 5