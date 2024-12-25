# Fastformer với bài toán News Recommendation (MINDs)
Phần này sẽ hướng dẫn cách sử dụng repo này để huấn luyện và nộp bài cho cuộc thi về MINDs trên Codalab

# Kết quả trên Codalab

| Độ đo     | AUC   | MRR   |nDCG@5    | nDCG@10   | 
|-----------|-------|-------|--------- |-----------|
| Kết quả   | 0.6813| 0.3331| 0.3617   | 0.4191    |

# Requirements
```
   python==3.10
   transformers==4.21.0
   scikit-learn==1.2.2
   tensorflow==2.16.1
```

# Chuẩn bị dữ liệu
Tải bộ dữ liệu MIND, sẽ nhận được 3 folder chính: ```MINDlarge_train```, ```MINDlarge_dev```, ```MINDlarge_test```. Đặt 3 folder này trong 1 folder to.

# Training
```
python train.py \
--pretreained_model bert \
--pretrained_model_path {path to ckpt of bert} \
--root_data_dir {path to MIND data}
--epochs 3 \
--num_hidden_layers 8 \
--world_size 4 \
--lr 1e-4 \
--warmup True \
--schedule_step 240000 \
--warmup_step 1000 \
--batch_size 16 \
--npratio 4 \
--savename rec_mind \
--news_dim 256
-- save_model_path ./saved-models/
```
Mô hình sau khi train sẽ được lưu trong 1 file checkpoint tại folder ```saved-models```

# Prediction
Dự đoán bằng checkpoint đã có được từ bước train và đưa kết quả dự đoán vào dạng zip
```
python submission.py \
--num_hidden_layers 8 \
--load_ckpt_name ./saved-models \
--batch_size 256 \
--news_dim 256
```
Sau khi thực hiện xong lệnh này sẽ tạo ra 1 file zip có tên 'prediction', có thể nộp file này trực tiếp lên cuộc thi MINDs trên Codalab
