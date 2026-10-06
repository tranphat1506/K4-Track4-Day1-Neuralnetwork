# Báo cáo Lab Day 1 — Báo cáo hoàn chỉnh

## 1. Thiết lập
- **Môi trường:** Google Colab, GPU T4, PyTorch 2.x
- **Dữ liệu:** Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val.
- **Model:** `M-base` (54→256→128→7, 47 879 tham số). Baseline: Cross-Entropy loss, SGD with momentum 0.9, lr=0.05, batch 512, 20 epochs, He initialization.
- **Mốc tham chiếu:** accuracy "đoán lớp đa số" trên val ≈ 0,4876.
- **Các chủ đề đã thử:** ☑ optimizer ☑ hyper-parameter ☑ kiến trúc (độ sâu/độ rộng)

## 2. Kiểm tra ban đầu và độ nhiễu
| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | ~ 2.37 (Hợp lý) |
| Quá khớp 20 mẫu: loss cuối | Giảm về gần 0 |
| Mọi tham số có gradient khác 0 | ☑ có |
| Baseline, số seed đã chạy | 3 seed (42, 43, 44) |
| Baseline: val acc (TB ± σ) | 0.903 ± 0.005 |
| Baseline: val macro-F1 (TB ± σ) | 0.845 ± 0.008 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0.016 (val macro-F1). 

## 3. Kết quả theo chủ đề

### 3.1 Bộ tối ưu hoá (Optimizer)
- **Dự đoán:** Adam và AdamW sẽ hội tụ nhanh hơn SGD vì khả năng điều chỉnh learning rate thích ứng cho từng tham số, đặc biệt hiệu quả trên dữ liệu mất cân bằng.
- **Kết quả:** (Dựa trên `adam-lr1e-3` và `adamw-lr1e-3`)
  - AdamW cho tốc độ hội tụ rất nhanh ở các epoch đầu và đạt Val Macro-F1 ổn định quanh mức 0.84 - 0.85, không bị overfit mạnh.
- **Giải thích:** SGD với momentum cần điều chỉnh learning rate rất cẩn thận. Adam/AdamW sử dụng các moment bậc 1 và bậc 2 để tự chuẩn hoá gradient, giúp thoát khỏi các vùng tối ưu cục bộ nhanh hơn ở những epoch đầu.

### 3.2 Kiến trúc (Độ rộng / Độ sâu)
- **Dự đoán:** Mạng sâu hơn (Deep) sẽ học được các đặc trưng phi tuyến tính phức tạp hơn, trong khi mạng rộng (Wide) sẽ có sức chứa lớn hơn cho tập dữ liệu gần nửa triệu mẫu.
- **Kết quả:** (`model-wide`, `model-deep`)
  - Model Wide (512 -> 256) tăng đáng kể số lượng tham số, có thể giúp Macro-F1 nhỉnh hơn một chút trên tập Val, nhưng kéo theo thời gian train lâu hơn.
- **Giải thích:** Dữ liệu có 54 đặc trưng (trong đó 44 cột là dạng one-hot), việc tăng số lượng node ẩn ở layer đầu (Wide) giúp kết hợp các biến one-hot này tốt hơn so với việc chồng quá nhiều layer (Deep) vốn dễ gây mất mát gradient nếu không có kỹ thuật ResNet.

## 4. Đánh giá cuối trên tập eval

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline | 42 | 0.845 | | |
| Cấu hình cuối cùng (AdamW) | 42 | 0.845+ | **0.8469** | **0.9026** |

- **Cấu hình cuối cùng:** Chọn `adamw-lr1e-3` vì nó cho thấy sự cân bằng tốt nhất giữa tốc độ huấn luyện, sự ổn định của gradient và điểm F1 cao.
- **Nhận xét:** Val và Eval rất sát nhau (0.845 vs 0.8469). Điều này chứng tỏ cách chia validation phân tầng 20% phản ánh cực kỳ chính xác phân phối của tập eval thực tế. Không hề có hiện tượng quá khớp (overfitting) nghiêm trọng lên tập val.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 | 42 368 | 0.902 | 0.905 | 0.903 |
| 1 | 56 661 | 0.913 | 0.924 | 0.918 |
| 2 |  7 151 | 0.880 | 0.874 | 0.877 |
| 3 |    549 | 0.780 | 0.841 | 0.809 |
| 4 |  1 899 | 0.730 | 0.765 | 0.747 |
| 5 |  3 473 | 0.828 | 0.702 | 0.760 |
| 6 |  4 102 | 0.958 | 0.868 | 0.911 |

- **Lớp khó nhất:** Là **Lớp 4** (F1 = 0.747) và **Lớp 5** (F1 = 0.760).
- **Phân tích nhầm lẫn:** 
  - Lớp 4 rất hay bị mô hình đoán nhầm sang **Lớp 1** (389 lần). 
  - Lớp 5 bị đoán nhầm nhiều sang **Lớp 2** (652 lần).
- **Lý giải:** Đây là hai lớp có Support (số lượng mẫu) khá nhỏ so với Lớp 0 và 1 (hàng chục nghìn mẫu). Dữ liệu bị mất cân bằng nghiêm trọng cộng với việc địa hình/đất đai của các loại rừng này có thể đan xen nhau (chứa các đặc trưng one-hot đất đai giống hệt nhau).
- **Cách cải thiện tương lai:** Sử dụng Focal Loss hoặc gán trọng số (Class Weights) lớn hơn cho lớp 4 và 5 để ép mô hình chú ý vào chúng.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng?** AdamW hội tụ ổn định và cho kết quả val tốt nhất trong thời gian ngắn nhất so với SGD.
6. **Câu hỏi bài học:** Một mạng có loss không giảm sau 2000 bước. 
   - **Phép thử 1:** Kiểm tra lại Loss bước 0 có gần bằng $-ln(1/C)$ không. Nếu không, lỗi nằm ở khâu Khởi tạo (Initialization).
   - **Phép thử 2:** Ép quá khớp (overfit) trên 1 batch rất nhỏ (ví dụ 20 mẫu). Nếu loss không thể giảm về 0, chắc chắn lỗi nằm ở kiến trúc mô hình (code sai shape, thiếu activation) hoặc pipeline huấn luyện (chưa `optimizer.step()`, chưa `zero_grad`).
   - **Phép thử 3:** Kiểm tra Learning Rate. Nếu overfit batch nhỏ thành công nhưng train thật vẫn không giảm, có thể learning rate quá nhỏ (mạng học quá chậm) hoặc quá to (gradient nổ, nhảy lung tung).

## 6. Hạn chế và điều bất ngờ
- **Bất ngờ:** Hiệu suất của MLP thuần tuý trên tập dữ liệu dạng bảng này rất ấn tượng (đạt >90% Acc), chứng tỏ các biến đặc trưng (one-hot loại đất) mang tín hiệu tuyến tính rất mạnh.
- **Hạn chế:** Chưa thử nghiệm kỹ các chiến thuật tuỳ chỉnh Learning Rate (như Cosine Annealing) hay các hàm loss trị mất cân bằng (Focal Loss). 

## 7. Phụ lục
- Đã nộp: `experiments.xlsx`, `REPORT.md`, `eval_result.json`, `predictions_eval.csv`, thư mục `code/`, thư mục `figures/`.
- Ước lượng thời gian chạy tổng cộng: ~ 10-15 phút trên T4 GPU.
