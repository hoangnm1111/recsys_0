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
Tải bộ dữ liệu MIND, sẽ nhận được 3 folder chính: ```MINDlarge_train```
