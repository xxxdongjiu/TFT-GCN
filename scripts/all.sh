export CUDA_VISIBLE_DEVICES=1

python -u run.py  --is_training 1  --root_path  ../dataset/SWaT   --data SWAT  --seq_len 100  --d_model 32  --anomaly_ratio 1  --batch_size 128  --train_epochs 3

python -u run.py  --is_training 1  --root_path /home/xusheng/dataset/SWaT   --data SWAT  --seq_len 100  --enc_in 51  --c_out 51  --d_model 32  --anomaly_ratio 1  --batch_size 128  --train_epochs 3

python -u run.py  --is_training 1  --root_path /home/xusheng/dataset/SMD   --data SMD  --seq_len 100  --enc_in 38  --c_out 38  --d_model 64  --anomaly_ratio 0.5  --batch_size 128  --train_epochs 10

python -u run.py  --is_training 1  --root_path /home/xusheng/dataset/PSM   --data PSM  --seq_len 100  --enc_in 25  --c_out 25  --d_model 64  --anomaly_ratio 1.5  --batch_size 256  --train_epochs 10

python -u run.py  --is_training 1  --root_path /home/xusheng/dataset/SMAP  --data SMAP  --seq_len 100  --enc_in 25  --c_out 25  --d_model 128  --anomaly_ratio 1  --batch_size 128  --train_epochs 10

python -u run.py  --is_training 1  --root_path /home/xusheng/dataset/MSL   --data MSL  --seq_len 100  --enc_in 55  --c_out 55  --d_model 8  --d_ff 32  --anomaly_ratio 1.5  --batch_size 256  --train_epochs 10 