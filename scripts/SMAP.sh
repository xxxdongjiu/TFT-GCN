python -u run.py  --is_training 1 \
    --root_path /home/xusheng/dataset/SMAP \
    --data SMAP \
    --seq_len 100 \
    --enc_in 25 \
    --c_out 25 \
    --d_model 128 \
    --anomaly_ratio 1 \
    --batch_size 128 \
    --train_epochs 5