python -u run.py  --is_training 1 \
    --root_path /home/xusheng/dataset/HAI \
    --data HAI \
    --seq_len 100 \
    --enc_in 59 \
    --c_out 59 \
    --d_model 32 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 5