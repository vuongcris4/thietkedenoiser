# Đề tài

## Video 7/1/2026

### Semi-supervised Semantic Segmentation (Seminar 7/1/2025)

#### Tổng quan (Overview)

##### Mục đích: Survey phương pháp để chọn Baseline

##### Phân loại chính

###### Model Probability (Cấu trúc mô hình)

###### Denoising Pseudo-labels (Xử lý nhãn nhiễu)

#### Mục đích

##### Survey & Review

###### Mục đích: Nhóm cần có cái nhìn toàn cảnh về các phương pháp hiện có trên thế giới (State-of-the-art) trong lĩnh vực Semi-supervised Semantic Segmentation.

###### Hành động: Phân tích sâu các bài báo (Papers) để hiểu rõ cơ chế hoạt động, ưu điểm và nhược điểm của từng nhóm phương pháp (Single-model vs Multi-model, Filter vs Update Label).

###### Câu nói chốt của người hướng dẫn: "Trước khi mình make decision (ra quyết định) là cái mình sẽ làm cái gì, thì mình phải nhìn người ta làm gì rồi, tại sao nó work, rồi mình mới propose cái của mình được."

##### Chọn ra "Baseline" (Mô hình cơ sở) tốt nhất

###### Mục đích: Tìm ra phương pháp nào đang có hiệu năng (Performance) tốt nhất và ổn định nhất để làm nền tảng.

###### Hành động: Yêu cầu các thành viên (Thiện, Danh) chạy thực nghiệm (Benchmark) lại các model đã survey.

###### Tư duy: "Muôn vạn lời lý thuyết cũng không bằng cái hiệu năng" - Nhóm không chọn phương pháp dựa trên lý thuyết hay, mà chọn cái nào chạy ra kết quả tốt nhất trên dữ liệu thực tế của họ để từ đó phát triển (customize) lên.

##### Sửa sai ("Troubleshooting") cho thực nghiệm hiện tại

###### Mục đích: Khắc phục lỗi thiết lập thí nghiệm (Experimental Setting) trong luận văn của thành viên tên Danh.

###### Vấn đề: Đang bị nhầm lẫn cách sử dụng dữ liệu Source Domain (có nhãn) và Target Domain (không nhãn/nhãn giả).

###### Giải pháp: Chốt lại kịch bản chạy thí nghiệm: Dùng 150 ảnh có nhãn cho Source, và đưa Pseudo-label vào Target (chế độ không nhãn) để chạy lại cho đúng chuẩn khoa học.

#### Nhóm 1: Training Frameworks (Model Probability)

##### Single-Model (Mô hình đơn)

###### ST (Self-Training)

####### Nguyên lý: Strong Data Augmentation

####### Kỹ thuật: Color Jitter, Grayscale, Cutout

####### Giả định: Nếu Model dự đoán nhất quán trên ảnh biến đổi mạnh -> Nhãn tin cậy

###### ST++ (Self-Training Plus Plus)

####### Cải tiến: Dựa trên độ ổn định qua các Checkpoints

####### Quy trình

######## So sánh dự đoán từ k checkpoints

######## Ổn định cao -> Tập tin cậy (Clean/Reliable) -> Học trước (Curriculum Learning)

######## Ổn định thấp -> Tập không tin cậy (Unreliable)

##### Multi-Model (Đa mô hình)

###### Mean Teacher

####### Cơ chế: Student update Teacher qua EMA (Exponential Moving Average)

###### CPS (Cross Pseudo Supervision)

####### Cơ chế: 2 mạng khởi tạo khác nhau

####### Hoạt động: Mạng A giám sát Mạng B và ngược lại (Cross Confidence)

#### Nhóm 2: Denoising Pseudo-labels (Xử lý nhiễu)

##### Label Update / Correction (Sửa nhãn)

###### Phương pháp Weakly Supervised (CAM)

####### Dùng Gradient CAM (Class Activation Maps)

####### Đánh giá: Không phù hợp với ảnh vệ tinh (Remote Sensing) do nhiều class/đối tượng phức tạp

###### Phương pháp Graph-based (2021/2022)

####### Kỹ thuật: Superpixels (Siêu điểm ảnh)

######## Gom nhóm pixel dựa trên màu sắc (RGB) và vị trí (XY)

####### Mô hình: Graph Attention Network (GCN)

####### Cơ chế Refine

######## Xác định nút "Clean" (Tin cậy) và nút "Uncertain" (Không chắc)

######## Dùng GCN để lan truyền nhãn từ Clean sang Uncertain dựa trên Feature Similarity

###### Phương pháp Dynamic Threshold

####### Vấn đề: Ngưỡng cố định (0.9) không tốt cho class khó

####### Giải pháp: Ngưỡng động (Dynamic) cho từng class

####### Phân loại pixel

######## High confidence -> Tin cậy (Positive)

######## Low confidence -> Chắc chắn sai (Negative)

######## Vùng giữa -> Uncertain

##### Traceable Training (2022)

###### Cơ chế: Image Aug vs Label Aug

####### Nhánh 1: Augment ảnh đầu vào -> Dự đoán

####### Nhánh 2: Dự đoán ảnh gốc -> Augment trên miền nhãn (Prediction domain)

####### Mục tiêu: Hai nhánh phải cho kết quả giống nhau (Consistency)

###### Kiến trúc: Sử dụng mạng phụ (Auxiliary Network) chia sẻ layer (n-1, n-2)

#### Thảo luận & Thực nghiệm (Experiments)

##### Sửa lỗi thiết lập (Experimental Setting)

###### Source Domain: Dữ liệu có nhãn (Labeled Data)

###### Target Domain: Dữ liệu không nhãn (Unlabeled Data)

####### Xử lý cũ: Đang dùng Pseudo-label làm Source (Sai)

####### Xử lý mới: Đưa Pseudo-label vào Target (Unsupervised mode)

##### Kế hoạch hành động (Action Items)

###### Chạy Benchmark các phương pháp (ST++, CPS, DAFormer...)

###### Mục tiêu: Tìm ra phương pháp có Performance tốt nhất trên dữ liệu thực tế

###### Viết báo cáo/Luận văn dựa trên kết quả thực nghiệm

## Video 8/1/2026

### Nhóm phương pháp

#### Vấn đề cốt lõi của Pseudo-labeling

##### Sự sai lệch giữa Độ tin cậy (Confidence) và Độ chính xác (Accuracy)

###### Model có thể rất tự tin (High confidence) nhưng vẫn dự đoán sai.

###### Cần cơ chế để lọc hoặc sửa các nhãn sai này thay vì tin tưởng mù quáng.

#### "Làm sao biết cái nhãn giả (Pseudo-label) mà Model vừa tạo ra là ĐÚNG hay SAI để quyết định có cho nó học hay không?"
  
  Nhóm Filter: Cái nào nghi ngờ sai thì vứt, không học.
  
  Nhóm Correction: Cái nào sai thì tìm cách sửa lại cho đúng rồi mới học.
  
  Nhóm Teacher-Student: Dùng một ông thầy (Teacher) điềm tĩnh hơn để sinh nhãn cho trò (Student) học, tránh việc trò tự biên tự diễn.

#### Nhóm phương pháp 1: Refine & Debias (Tinh chỉnh và Khử thiên kiến)

##### Xử lý Bias do dữ liệu (Data Bias) - Bài báo ICCV 2021

###### Vấn đề: Mất cân bằng dữ liệu (Long-tail), các class khó ít được gán nhãn đúng.

###### Giải pháp: Dynamic Threshold (Ngưỡng động).

####### Thay vì dùng ngưỡng cố định, dùng ngưỡng riêng cho từng class.

####### Dựa trên thống kê số lượng pixel/mẫu của từng class.

###### Kỹ thuật: Re-sampling (Lấy mẫu lại) để cân bằng phân phối giữa các class.

##### Xử lý Bias do mô hình (Model Bias) - Bài báo NeurIPS 2022

###### Vấn đề: Model bị Overfitting vào tập dữ liệu có nhãn (Labelled set) quá nhỏ.

###### Hậu quả: Đường phân lớp bị lệch về phía dữ liệu có nhãn, bỏ qua cấu trúc thực tế của dữ liệu không nhãn.

###### Giải pháp: Decoupled Head (Tách đầu phân loại).

###### Kỹ thuật: Feature Clustering.

####### Dùng loss function để kéo đặc trưng (feature) của dữ liệu không nhãn về gần cụm của dữ liệu có nhãn.

####### Tăng tính phân biệt (discrimination) trong không gian đặc trưng.

#### Nhóm phương pháp 2: Filter Only (Lọc bỏ mẫu sai)

##### Mở rộng CPS (Cross Pseudo Supervision) với Discriminator

###### Ý tưởng: Dùng mạng GAN (Discriminator) để đánh giá chất lượng nhãn giả.

###### Cơ chế hoạt động:

####### Input: Cặp (Ảnh đầu vào + Bản đồ dự đoán/Nhãn).

####### Discriminator phân biệt:

######## Real: Ảnh thật + Nhãn thật (Ground Truth).

######## Fake: Ảnh thật + Nhãn giả (Pseudo-label).

###### Kết quả: Chỉ dùng các pixel/vùng mà Discriminator chấm là "Real" (đúng) để train lại model.

##### Curriculum Learning (Học theo lộ trình) - Bài báo 2023

###### Nguyên lý: "Học dễ trước, học khó sau".

###### Kỹ thuật: Pruned Teacher (Giáo viên bị cắt tỉa).

####### So sánh Feature của Teacher gốc (Full) và Teacher bị cắt bớt trọng số (Pruned).

####### Dùng Cosine Similarity để so sánh.

###### Logic đánh giá độ khó:

####### Nếu cắt bớt model mà Feature vẫn giống nhau -> Mẫu dễ/Tin cậy -> Dùng để học ngay.

####### Nếu Feature thay đổi nhiều -> Mẫu khó/Không tin cậy -> Bỏ qua hoặc học sau.

##### GTA (Generative Teaching Assistant)

###### Cơ chế: Thêm một module "Trợ giảng" (Assistant) vào giữa Teacher và Student.

###### Vai trò:

####### Assistant học từ nhãn nhiễu (Pseudo-label).

####### Chuyển giao tri thức (Knowledge Transfer) sang Student.

###### Mục đích: Giúp Student không bị "học vẹt" (overfit) vào tập Ground Truth hạn hẹp.

#### Nhóm phương pháp 3: Optimization (Tối ưu hóa)

##### Dual Teacher (Hai giáo viên)

###### Cơ chế: Dùng 2 Teacher model khác nhau để sinh nhãn giả.

###### Cách dùng: Lấy trung bình cộng (Average) dự đoán của 2 teacher để tăng độ ổn định.

##### Adversarial Noise (Nhiễu đối kháng)

###### Khác biệt với nhiễu Gaussian (ngẫu nhiên): Nhiễu này được sinh ra có chủ đích để thay đổi kết quả dự đoán nhiều nhất.

###### Mục đích: Tấn công vào các điểm yếu (boundary) của model.

###### Tác dụng: Giúp model mạnh mẽ hơn (robust) với các mẫu nằm ở ranh giới phân lớp.

##### Multi-model Strategy (TP 2023)

###### Sử dụng 2 nhánh mạng song song.

###### Xử lý dựa trên sự Đồng thuận (Agreement):

####### Nếu 2 nhánh cùng dự đoán giống nhau -> Tin cậy cao -> Dùng Intersection Loss.

###### Xử lý dựa trên sự Bất đồng (Disagreement):

####### Nếu 2 nhánh dự đoán khác nhau -> Dùng "Agreement Matrix" (dạng Confusion Matrix).

####### So sánh độ tin cậy toàn cục của các class để quyết định tin nhánh nào.

####### Dùng Union Loss cho trường hợp này.

## Video 9/1/2026

### Nghiên cứu Semi-supervised Segmentation cho ảnh vệ tinh

#### Bối cảnh & Vấn đề (Problem Definition)

##### Sự khác biệt với dữ liệu ảnh thường (như PASCAL VOC)

###### Hình dạng (Shape)

####### Ảnh thường: Đối tượng (xe, người) có hình dạng cụ thể, rõ ràng.

####### Ảnh vệ tinh: Đối tượng (các loại đất, rừng) không có hình dạng cố định (amorphous).

###### Ranh giới (Boundary)

####### Ảnh thường: Ranh giới vật thể tách biệt rõ với nền.

####### Ảnh vệ tinh: Ranh giới giữa các loại đất (ví dụ: đất trống vs. đất quy hoạch) rất mờ nhạt và khó xác định.

###### Phân bố lớp (Class Distribution)

####### Ảnh thường: Số lượng class trong một ảnh ít.

####### Ảnh vệ tinh: Một ảnh chứa rất nhiều class phức tạp xen kẽ.

##### Thách thức chính (Key Challenges)

###### Visual Similarity (Sự tương đồng thị giác)

####### Các class khác nhau (về bản chất) nhưng quan sát hình ảnh lại rất giống nhau.

####### Ví dụ: Đất đã quy hoạch (nhưng chưa xây) nhìn rất giống đất hoang.

###### Sự thất bại của các phương pháp cũ

####### Các phương pháp hiện tại thường dựa vào giả định "hình dạng giống nhau thì nhãn giống nhau" -> Sai với ảnh vệ tinh.

#### Các đề xuất giải pháp (Proposed Methods)

##### 1. Cải tiến Label Correction (Sửa nhãn nhiễu)

###### Vấn đề cũ

####### Dùng "Query Image" (ảnh tham chiếu) để sửa nhãn cho ảnh không nhãn.

####### Dựa trên sự tương đồng về hình dạng (Shape-based) -> Không hiệu quả.

###### Giải pháp mới (Pixel-level Query)

####### Thay vì query cả ảnh, hãy query ở cấp độ Pixel hoặc Super-pixel.

####### Lấy các Pixel tin cậy từ tập dữ liệu có nhãn (Labeled Data) làm mỏ neo (anchor) để sửa nhãn.

##### 2. Prototype / Contrastive Learning

###### Xây dựng các đặc trưng mẫu (Prototype) cho từng class.

###### Dùng Contrastive Learning để kéo các đặc trưng của ảnh về gần với Prototype của class đó thay vì dựa vào hình dạng.

##### 3. Boundary Loss tùy chỉnh

###### Sử dụng Super-pixel để tìm các ranh giới tự nhiên (low-level features).

###### Nguyên lý: Nếu Super-pixel có ranh giới, thì kết quả Segmentation cũng nên có ranh giới tại đó.

###### Chấp nhận sai số một chiều: Super-pixel có biên thì Segmentation phải có, nhưng ngược lại thì có thể bỏ qua.

##### 4. Diffusion Model cho Denoising (Ý tưởng mới)

###### Cách tiếp cận đột phá (Out-of-the-box)

####### Coi các nhãn dự đoán sai (pseudo-label noise) là một dạng tín hiệu nhiễu.

####### Thay vì dùng GAN để phát hiện nhiễu (như cũ), dùng Diffusion Model để "khử nhiễu" trực tiếp.

###### Quy trình

####### Train model Diffusion để biến đổi từ "nhãn nhiễu" sang "nhãn sạch".

#### Kế hoạch thực hiện (Action Plan)

##### Thiết lập Baseline

###### Chạy lại thí nghiệm trên các model hiện có để xác định nền tảng so sánh.

###### Lưu ý việc setup lại tập dữ liệu label/unlabel cho đúng chuẩn.

##### Chứng minh luận điểm (Validation)

###### Chuẩn bị slide/dẫn chứng cụ thể về việc "Shape" và "Boundary" là vấn đề lớn trong dataset vệ tinh.

###### Chứng minh bằng trực quan: Zoom vào ảnh để thấy sự giống nhau gây nhầm lẫn giữa các class.

##### Phát triển thuật toán

###### Tập trung vào hướng Pixel-level Query hoặc Diffusion Denoising.

## Cần làm Denoiser 

### Tổng quan & Mục tiêu (Overview)

#### Mục đích chính

##### Survey phương pháp hiện có (SOTA)

##### Chọn Baseline tốt nhất dựa trên hiệu năng thực tế

##### Thiết kế bộ Denoiser (Xử lý nhãn nhiễu)

#### Tư duy cốt lõi

##### Performance > Theory (Hiệu năng quan trọng hơn lý thuyết)

##### Không chọn phương pháp dựa trên giả định sai (Shape-based)

##### Phải hiểu tại sao họ làm vậy trước khi propose cái mới

### Vấn đề cốt lõi (Problem Definition)

#### Nghịch lý Pseudo-label

##### High Confidence ≠ High Accuracy (Tự tin cao chưa chắc đúng)

##### Cần cơ chế lọc/sửa nhãn thay vì tin tưởng mù quáng

#### Thách thức đặc thù của Ảnh vệ tinh (Remote Sensing)

##### Hình dạng (Shape)

###### Đối tượng vô định hình (Amorphous)

###### Không có hình dạng cụ thể như ảnh thường (xe, người)

##### Ranh giới (Boundary)

###### Ranh giới mờ nhạt (Vd: Đất trống vs Đất quy hoạch)

###### Khó tách biệt so với ảnh PASCAL VOC

##### Độ tương đồng (Visual Similarity)

###### Các class khác nhau nhưng nhìn rất giống nhau

###### Dễ gây nhầm lẫn cho Model

### Thiết lập thực nghiệm (Experimental Setting)

#### Sửa lỗi quy trình (Fixing Mistakes)

##### Lỗi cũ: Dùng Pseudo-label làm Source (Sai)

##### Quy trình chuẩn:

###### Source Domain: 150 ảnh có nhãn (Labeled Data)

###### Target Domain: Dữ liệu không nhãn + Pseudo-label (Unsupervised Mode)

#### Hành động (Action)

##### Benchmark lại toàn bộ các model (ST++, CPS, DAFormer...)

##### Chọn ra model có chỉ số tốt nhất làm Baseline

### Cơ sở lý thuyết: Các nhóm phương pháp (Background)

#### Training Frameworks (Cấu trúc)

##### Single-Model

###### Self-Training (ST): Strong Augmentation (Color Jitter, Cutout)

###### ST++: Dựa trên độ ổn định qua các Checkpoints

##### Multi-Model

###### Mean Teacher: EMA update

###### CPS (Cross Pseudo Supervision): 2 mạng giám sát chéo nhau

#### Denoising Strategies (Chiến lược khử nhiễu hiện có)

##### Label Update/Correction

###### Weakly Supervised (CAM): Không phù hợp ảnh vệ tinh

###### Graph-based (GCN): Dùng Superpixels lan truyền nhãn từ Clean -> Uncertain

##### Filter Only

###### Dynamic Threshold: Ngưỡng động cho từng class (xử lý Long-tail)

###### Discriminator (GAN): Phân biệt nhãn Real vs Fake

##### Optimization

###### Dual Teacher: Lấy trung bình dự đoán

###### Adversarial Noise: Tấn công vào boundary để tăng độ bền vững

### Đề xuất: Thiết kế bộ Denoiser (Proposed Solution)

#### Nguyên lý thiết kế

##### Không dựa vào Shape (Hình dạng) để sửa nhãn

##### Tập trung vào Pixel-level và Feature Consistency

##### Xử lý trực tiếp nhiễu (Noise) trong Pseudo-label

#### Giải pháp 1: Pixel-level Query & Prototype

##### Thay đổi cách Query

###### Cũ: Image-level (Dựa trên Shape toàn ảnh) -> Sai

###### Mới: Pixel-level / Super-pixel level

##### Cơ chế

###### Dùng pixel tin cậy từ Source (Labeled) làm "Neo" (Anchor)

###### Sử dụng Contrastive Learning để kéo Feature ảnh không nhãn về gần Prototype của Class

#### Giải pháp 2: Boundary Refinement (Tinh chỉnh biên)

##### Công cụ: Super-pixels

##### Nguyên lý

###### Tìm các ranh giới tự nhiên (low-level features)

###### Nếu Super-pixel có biên -> Segmentation mask phải có biên

###### Chấp nhận sai số một chiều (Super-pixel có -> Mask có)

#### Giải pháp 3: Generative Denoising (Đột phá)

##### Công nghệ: Diffusion Model

##### Ý tưởng

###### Coi "Pseudo-label noise" là một dạng tín hiệu nhiễu

###### Train Diffusion Model để chuyển đổi: Nhãn nhiễu -> Nhãn sạch

##### Ưu điểm: Khả năng tái tạo cấu trúc tốt hơn GAN

##### chi tiết

###### Giải pháp 3: Generative Denoising (Diffusion Model)

####### Tư duy cốt lõi (Core Concept)

######## Góc nhìn cũ (Traditional/Filter)

######### Coi nhãn là: Đối tượng logic (Đúng hoặc Sai)

######### Hành động: Sàng lọc (Filtering)

######### Nhược điểm: Bỏ phí dữ liệu nếu nhãn chỉ sai một chút

######## Góc nhìn mới (Generative/Restore)

######### Coi nhãn là: Tín hiệu hình ảnh (Image Signal)

######### Vấn đề: Pseudo-label là một bức ảnh bị "mờ" hoặc "nhiễu"

######### Hành động: Phục chế (Restoration)

######### Mục tiêu: Biến nhãn sai thành nhãn đúng thay vì vứt bỏ

####### Cơ chế hoạt động (Mechanism)

######## Quy trình thuận (Forward Process - Training)

######### Input: Nhãn thật (Ground Truth - GT)

######### Hành động: Cố tình thêm nhiễu (Gaussian noise) vào GT theo thời gian t

######### Kết quả: Biến GT sạch thành một bản đồ nhiễu (Mô phỏng Pseudo-label lỗi)

######## Quy trình nghịch (Reverse Process - Denoising)

######### Mục tiêu: Học cách đảo ngược quá trình thêm nhiễu

######### Input: Bản đồ nhiễu (Noisy Mask)

######### Điều kiện (Conditioning): Ảnh vệ tinh gốc (Raw Image)

######### Logic: "Nhìn vào ảnh vệ tinh để biết cách sửa lại bản đồ nhãn đang bị sai"

######### Output: Nhãn sạch (Refined Label)

####### Tại sao vượt trội cho ảnh vệ tinh? (Why Remote Sensing?)

######## Khắc phục Ranh giới mờ (Ambiguous Boundaries)

######### Vấn đề: Ranh giới giữa các loại đất thường không rõ ràng

######### Diffusion: Tạo ra các đường ranh giới mượt mà (smooth) và liên tục, tránh hiện tượng răng cưa

######## Khắc phục Nhiễu lốm đốm (Salt-and-pepper Noise)

######### Vấn đề: Model thường dự đoán sai các pixel nhỏ lẻ (vd: điểm nước giữa rừng)

######### Diffusion: Có tính chất làm mịn (smoothing), tự động đồng nhất các vùng nhỏ vào bối cảnh lớn

######## Bảo toàn Cấu trúc không gian (Spatial Consistency)

######### Vấn đề: GAN đôi khi tạo ra hình dạng phi vật lý

######### Diffusion: Học được phân phối hình dạng (Shape Prior)

######### Ví dụ: Hiểu rằng "con đường" phải là một dải liền mạch, không được đứt đoạn vô lý

####### Quy trình thực hiện (Implementation Workflow)

######## Bước 1: Initial Pseudo-labeling

######### Train model cơ bản (UNet/SegFormer) trên tập có nhãn

######### Sinh nhãn giả (Noisy Pseudo-labels) cho tập không nhãn

######## Bước 2: Train Diffusion Denoiser

######### Sử dụng tập dữ liệu có nhãn (Labeled Set)

######### Học mô hình xác suất: P(Clean Label | Noisy Label, Satellite Image)

######## Bước 3: Label Refinement

######### Dùng Denoiser đã học để sửa lỗi cho tập Pseudo-labels ở Bước 1

######### Kết quả: Tập Pseudo-labels chất lượng cao (High-quality)

######## Bước 4: Final Segmentation Training

######### Dùng tập nhãn đã làm sạch để train model phân lớp cuối cùng

######### Đạt hiệu năng vượt trội so với Baseline

####### Từ khóa tham khảo (Keywords)

######## Denoising Diffusion Probabilistic Models (DDPM)

######## Conditional Diffusion for Segmentation

######## Label Refinement via Generative Models

######## SegDiff

#### Giải pháp 4: Feature-based Filtering

##### Discriminator mở rộng

###### Input: Cặp (Ảnh gốc + Pseudo-label)

###### Loại bỏ các vùng mà Discriminator đánh giá là "Fake"

##### Decoupled Head

###### Tách đầu phân loại để xử lý Model Bias

###### Gom cụm feature (Clustering) dữ liệu không nhãn về phía có nhãn

### Kế hoạch hành động (Action Plan)

#### Bước 1: Chạy lại Benchmark (Sửa setup Source/Target)

#### Bước 2: Chứng minh vấn đề (Validation)

##### Chuẩn bị hình ảnh minh họa sự thất bại của Shape-based trong ảnh vệ tinh

##### Zoom vào các vùng ranh giới mờ nhạt

#### Bước 3: Phát triển Denoiser

##### Ưu tiên hướng: Pixel-level Query hoặc Diffusion Denoising

#### Bước 4: Viết báo cáo/Luận văn

## Cuộc họp Nghiên cứu Segmentation (Tóm tắt)

### 1. Bối cảnh & Vấn đề (Context & Problem)

#### Khác biệt về dữ liệu

##### Bộ dữ liệu thông thường (BC VOC): Một ảnh thường chỉ chứa 1-2 đối tượng chính.

##### Bộ dữ liệu vệ tinh (OpenSMap): Một ảnh chứa nhiều class cùng lúc (đa đối tượng).

#### Hạn chế của phương pháp cũ (Previous Works)

##### Thường sử dụng Image-level Query (Prototype đại diện cho cả ảnh).

##### Không hiệu quả với ảnh vệ tinh do tính chất đa lớp phức tạp trong một ảnh.

#### Thách thức về sự đồng xuất hiện (Co-occurrence)

##### Các class thường xuất hiện cùng nhau (ví dụ: có nước thì có cây, đất cằn cỗi đi với đồng cỏ).

##### Gây khó khăn cho việc tách biệt đối tượng dựa trên ngữ cảnh đơn thuần.

### 2. Giải pháp đề xuất (Proposed Method)

#### Ý tưởng cốt lõi: Pixel-level Query

##### Thay vì query ở mức ảnh (Image-level), chuyển sang query ở mức điểm ảnh (Pixel-level).

##### Cho phép nắm bắt sự đa dạng nội tại của class (Intra-class variation).

##### Ví dụ: Phân biệt được đặc trưng "nhà ở nông thôn" khác với "nhà ở thành thị" dù cùng là class "nhà".

#### Cơ chế Anchor Pixel (Prototype)

##### Sử dụng các "Anchor Pixels" như các tâm cụm (Cluster Centers) có thể học được (Learnable).

##### Không cố định số lượng prototype cứng nhắc cho mỗi class, để mô hình tự học sự phân bố.

##### Prototype được lưu và cập nhật bên trong model (như tham số) thay vì Memory Bank bên ngoài phức tạp.

### 3. Kiến trúc & Hàm mất mát (Loss Functions)

#### Backbone

##### Sử dụng SegFormer hoặc ResNet-101.

##### Đề xuất loại bỏ Projector thừa ở nhánh Segmentation chính để tận dụng feature trực tiếp.

#### Các loại Loss Function

##### Segmentation Loss: Hàm mục tiêu chính để phân lớp có giám sát.

##### Contrastive Loss: Đảm bảo feature không bị suy biến (chiếu về 0) và tăng khoảng cách giữa các feature khác loại.

##### Diversity Loss: Ép các Anchor (Prototype) phải xa nhau để tăng độ đa dạng, bao phủ không gian dữ liệu tốt hơn.

##### Boundary Loss: Sử dụng Superpixel để cải thiện độ chính xác tại biên (boundary) của đối tượng.

### 4. Quy trình huấn luyện (Framework)

#### Giai đoạn 1 (Stage 1)

##### Train model với Pixel-level Anchor + các loại Loss đã định nghĩa.

#### Giai đoạn 2 (Self-training/Refinement)

##### Sử dụng Pseudo-label sinh ra từ Stage 1 để train/refine lại (Teacher-Student model).

##### Tham khảo quy trình của framework "FITS".

### 5. Phân công nhiệm vụ (Assignments)

#### Linh

##### Implement core method: Pixel-level Anchor, Contrastive/Diversity Loss.

##### Thời gian dự kiến: ~1 tuần.

#### Thiện

##### Xây dựng Training Framework tổng thể.

##### Tích hợp Stage 1 vào luồng Self-training.

##### Nghiên cứu kỹ paper/code của FITS.

#### Danh

##### Chuẩn bị Data (OpenSMap, Inria...) và Data Loader chuẩn.

##### Chạy thí nghiệm (Experiments) và Debug sau khi Linh code xong.

##### Chuẩn bị slide báo cáo chi tiết về bài báo FITS.

#### Vĩ & Vương

##### Nghiên cứu hướng tiếp cận song song: Denoising.

##### Ý tưởng: Dùng Conditional Auto-encoder/Diffusion để khử nhiễu từ input (RGB + Pseudo-label).

#### Huấn

##### Chạy thực nghiệm Baseline trên Backbone SegFormer để so sánh hiệu năng.

### Link YouTube

# Phát triển Denoiser

## Kiến trúc model latent-diffusion

###  

## Các bước chạy thí nghiệm

### B1: Làm nhiễu segmentic map CxWxH

#### Nhân với ma trận 0..1

### B2: Bỏ data ở B1 vào x của latent diffusion

#### Conditioning là input của data ImageEarthdatset

## Lên plan clone dataset và Diffusion về chạy thí nghiệm

### Cho tui con số cấu hình cần thiết để chạy thí nghiệm này

# Dataset

## OpenEarthMap: Benchmark Dataset cho Bản đồ Lớp phủ Đất Độ Phân giải Cao Toàn cầu

### 1. Giới thiệu

#### Vấn đề hiện tại

##### Thiếu dữ liệu mẫu ở các nước đang phát triển

##### Chất lượng annotation thô sơ trong các benchmark hiện có

##### Hình ảnh vệ tinh thương mại không thể phân phối lại

#### Mục tiêu

##### Cung cấp bản đồ tự động cho mọi người

##### Đa dạng địa lý và chất lượng annotation cao

### 2. Bộ dữ liệu

#### Thống kê: 5000 ảnh, 97 vùng, 44 quốc gia, 6 lục địa

#### Độ phân giải: 0.25-0.5m GSD, 1024×1024 pixels

#### 8 Lớp phân loại

##### Bareland: 1.5%

##### Rangeland: 22.9%

##### Developed: 16.1%

##### Road: 6.7%

##### Tree: 20.2%

##### Water: 3.3%

##### Agriculture: 13.7%

##### Building: 15.6%

#### Chia dữ liệu: Train/Val/Test = 3000/500/1500

### 3. Semantic Segmentation

#### CNN-based

##### U-Net + VGG-11: 64.97% mIoU

##### U-Net + ResNet-34: 65.43% mIoU

##### U-Net + EfficientNet-B4: 68.20% mIoU ⭐

##### DeepLabV3 + ResNet-50: 63.16% mIoU

##### HRNet + W48: 63.07% mIoU

#### Transformer-based

##### SegFormer + MiT-B5: 64.01% mIoU

##### SETR PUP + ViT-L: 61.40% mIoU

##### UPerNet + ViT: 60.52% mIoU

##### UPerNet + Swin-B: 66.09% mIoU ⭐

##### UPerNet + Twins: 64.05% mIoU

##### UPerNet + ConvNeXt: 61.93% mIoU

##### K-Net + Swin-B: 66.11% mIoU

#### Hybrid

##### U-NetFormer + ResNeXt101: 68.37% mIoU

##### FT-U-NetFormer + Swin-B: 69.13% mIoU ⭐⭐ (Tốt nhất)

#### IoU theo lớp (FT-U-NetFormer)

##### Water: 87.44% (cao nhất)

##### Building: 80.29%

##### Agriculture: 77.50%

##### Tree: 73.33%

##### Road: 65.85%

##### Rangeland: 60.84%

##### Developed: 57.58%

##### Bareland: 50.19% (thấp nhất)

### 4. Neural Architecture Search (NAS)

#### SparseMask

##### Trial 1: 60.21% mIoU, 2.96M params

##### Trial 2: 60.00% mIoU, 3.10M params

##### Ưu: Độ chính xác cao với TTA

#### FasterSeg

##### Trial 1: 58.35% mIoU, 2.23M params

##### Trial 2: 59.41% mIoU, 3.47M params

##### Ưu: Tốc độ cao (143-171ms), phù hợp real-time

#### So sánh: <4M params cạnh tranh với UPerNet-ViT 144M params

### 5. Limited Labels (10% Training Data)

#### U-Net-EfficientNet-B4: 60.18% mIoU ⭐

#### K-Net: 54.33% mIoU

#### UPerNet-Swin-B: 52.39% mIoU

#### SegFormer: 51.61% mIoU

#### Kết luận: CNN vượt Transformer 6-15% khi dữ liệu ít

### 6. UDA Regional-level (DeepLabV2-based)

#### Oracle: 54.65% mIoU

#### Source only: 50.01% mIoU

#### Adversarial Training

##### TransNorm: 51.01% mIoU

##### CLAN: 48.85% mIoU

##### MCD: 47.36% mIoU

##### AdaptSeg: 45.60% mIoU

##### FADA: 44.35% mIoU

##### PyCDA: 37.89% mIoU

#### Self-Training

##### IAST: 53.46% mIoU

##### CBST: 52.01% mIoU

#### Kết luận: Self-Training > Adversarial Training

### 7. UDA Regional-level (SegFormer-based)

#### Oracle: 64.99% mIoU

#### DAFormer: 62.35% mIoU ⭐⭐ (Tốt nhất UDA)

#### Source only: 59.32% mIoU

#### Kết luận: DAFormer hiệu quả nhờ SegFormer + Self-Training

### 8. UDA Continent-wise

#### Domain gap nhỏ

##### Europe → North America

##### Asia → North America

#### Domain gap lớn

##### Africa → Europe

##### North America → Africa

#### Oceania: Dữ liệu ít → kết quả thấp nhất

#### DAFormer tốt hơn SegFormer: 20/30 trường hợp

### 9. Cross-Dataset Evaluation

#### Pre-train OpenEarthMap

##### Chesapeake Bay: 56.91% mIoU

##### Fine-tune DeepGlobe: IoU ban đầu +4%

#### Pre-train LoveDA

##### Chesapeake Bay: 46.04% mIoU

#### Kết luận: OpenEarthMap pre-train vượt trội ~11%

### 10. Đóng góp chính

#### Đa dạng địa lý toàn cầu

#### Annotation chi tiết không gian (2.2M segments)

#### Benchmark đa task: Segmentation, UDA, NAS

#### Pre-training hiệu quả

### 11. Hạn chế & Hướng nghiên cứu

#### ViT cần nhiều dữ liệu hơn CNN

#### UDA theo lục địa còn thách thức

#### Cần nghiên cứu data augmentation

### 12. Tài nguyên

#### https://open-earth-map.org

# Diffusion Model

## Diffusion Models Research

### 1. DDPM - Denoising Diffusion Probabilistic Models (2020)

#### Tác giả

##### Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)

#### Hội nghị

##### NeurIPS 2020

#### Ý tưởng chính

##### Mô hình biến thể ẩn (latent variable model) dựa trên chuỗi Markov

##### Học reverse process để khử nhiễu từ noise → data

##### Kết nối với denoising score matching và Langevin dynamics

#### Quá trình Forward (Diffusion)

##### Thêm nhiễu Gaussian dần vào data qua T bước

##### q(xt|xt-1) = N(xt; √(1-βt)xt-1, βtI)

##### Có closed-form: q(xt|x0) = N(xt; √ᾱt·x0, (1-ᾱt)I)

##### ᾱt = ∏αs với αt = 1-βt

#### Quá trình Reverse (Generative)

##### Học pθ(xt-1|xt) để đảo ngược forward process

##### pθ(xt-1|xt) = N(xt-1; μθ(xt,t), Σθ(xt,t))

##### Bắt đầu từ xT ~ N(0,I), sample ngược về x0

#### Parameterization ε-prediction

##### Thay vì predict μ, mô hình predict noise ε

##### μθ = (1/√αt)(xt - βt/√(1-ᾱt)·εθ(xt,t))

##### Liên hệ với denoising score matching

#### Hàm Loss

##### Variational bound: L = LT + Σt Lt-1 + L0

##### Simplified: Lsimple = E[||ε - εθ(√ᾱt·x0 + √(1-ᾱt)·ε, t)||²]

##### Down-weight loss ở t nhỏ → focus vào denoising khó

#### Kiến trúc mạng

##### U-Net backbone từ PixelCNN++

##### Group Normalization

##### Self-attention ở resolution 16×16

##### Time embedding: Transformer sinusoidal position

#### Kết quả

##### CIFAR10: IS=9.46, FID=3.17 (state-of-the-art)

##### LSUN 256×256: chất lượng tương đương ProgressiveGAN

#### Đóng góp chính

##### Chứng minh diffusion models có thể sinh ảnh chất lượng cao

##### Kết nối lý thuyết với score matching và Langevin dynamics

##### Phân tích progressive lossy compression

#### Hạn chế

##### Sampling chậm (T=1000 bước)

##### Log-likelihood không cạnh tranh với các mô hình khác

### 2. DDIM - Denoising Diffusion Implicit Models (2021)

#### Tác giả

##### Jiaming Song, Chenlin Meng, Stefano Ermon (Stanford)

#### Hội nghị

##### ICLR 2021

#### Ý tưởng chính

##### Tổng quát hóa DDPM với non-Markovian forward process

##### Tạo generative process nhanh hơn 10-50× so với DDPM

##### Implicit probabilistic model (deterministic khi σ=0)

#### Non-Markovian Forward Process

##### Giữ nguyên marginal q(xt|x0) như DDPM

##### Định nghĩa qσ(xt-1|xt,x0) với variance σ có thể điều chỉnh

##### Nhiều forward processes khác nhau → cùng training objective

#### Generative Process

##### xt-1 = √αt-1·f(xt) + √(1-αt-1-σ²)·εθ(xt) + σ·ε

##### f(xt) = (xt - √(1-αt)·εθ(xt))/√αt (predicted x0)

##### σ=0: deterministic (DDIM)

##### σ=√((1-αt-1)/(1-αt))·√(1-αt/αt-1): stochastic (DDPM)

#### Shared Training Objective

##### Theorem 1: Jσ = Lγ + C (với mọi σ > 0)

##### Có thể dùng model đã train với DDPM objective

##### Không cần retrain khi thay đổi σ

#### Accelerated Sampling

##### Chọn subsequence τ của [1,...,T]

##### Sample theo trajectory ngắn hơn (S << T steps)

##### Vẫn dùng cùng pretrained model

#### Liên hệ với Neural ODEs

##### DDIM tương đương Euler integration của ODE

##### d͞x(t) = εθ(͞x(t)/√(σ²+1))·dσ(t)

##### Có thể encode x0 → xT và reconstruct

#### Sample Consistency

##### Cùng xT → samples tương tự bất kể trajectory length

##### High-level features được encode trong xT

#### Kết quả

##### 10-50× nhanh hơn DDPM với chất lượng tương đương

##### CIFAR10: FID=4.16 với 100 steps (vs 1000 steps DDPM)

##### CelebA: FID=6.53 với 100 steps

#### Ưu điểm so với DDPM

##### Sampling nhanh hơn đáng kể

##### Deterministic generation → reproducible

##### Semantic interpolation trong latent space

##### Có thể encode và reconstruct images

#### Đóng góp chính

##### Non-Markovian forward process framework

##### Accelerated sampling không cần retrain

##### Kết nối với Neural ODEs

### 3. Latent Diffusion Models - LDM (2022)

#### Tác giả

##### Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer

#### Tổ chức

##### LMU Munich, Heidelberg University, Runway ML

#### Ý tưởng chính

##### Train diffusion models trong latent space thay vì pixel space

##### Tách perceptual compression và semantic compression

##### Giảm chi phí training và inference đáng kể

#### Hai giai đoạn training

##### Stage 1: Train autoencoder (E, D) trên image space

###### Perceptual loss + adversarial objective

###### KL-regularization hoặc VQ-regularization

###### Downsampling factor f = 4, 8, 16...

##### Stage 2: Train diffusion model trên latent space z = E(x)

###### LLDM = E[||ε - εθ(zt, t)||²]

###### Computationally efficient

###### Reuse autoencoder cho nhiều diffusion models

#### Perceptual Compression

##### Autoencoder loại bỏ high-frequency imperceptible details

##### Latent space có dimensionality thấp hơn nhiều

##### Reconstruction vẫn faithful về perceptual quality

#### Latent Diffusion

##### Forward process trong latent space z

##### zt = √ᾱt·z + √(1-ᾱt)·ε

##### Decoder D chuyển samples z0 → images x̃

#### Conditioning Mechanisms

##### Concatenation: cho spatial conditioning (inpainting, super-resolution)

##### Cross-attention: cho text, class labels, layouts

###### Q = W_Q·φ(zt), K = W_K·τθ(y), V = W_V·τθ(y)

###### Attention(Q,K,V) = softmax(QK^T/√d)·V

##### τθ: domain-specific encoder (e.g., BERT for text)

#### Trade-off Analysis

##### LDM-1 (pixel): slow training, expensive

##### LDM-4, LDM-8: optimal balance

##### LDM-16, LDM-32: too much compression, quality loss

#### Ứng dụng

##### Unconditional image synthesis

###### CelebA-HQ: FID=5.11 (SOTA)

###### FFHQ, LSUN-Churches, LSUN-Bedrooms

##### Text-to-image synthesis

###### Training trên LAION-400M

###### 1.45B parameters

###### Classifier-free guidance

##### Class-conditional ImageNet

###### FID=3.60 với LDM-4-G

###### Vượt ADM với ít parameters hơn

##### Super-resolution

###### 4× upscaling: FID=2.4

###### Concatenation conditioning

##### Inpainting

###### SOTA trên Places dataset

###### Convolution-like application cho high-resolution

##### Layout-to-image

###### Conditioning on bounding boxes

###### OpenImages, COCO

#### Kết quả Training Efficiency

##### 2.7× speedup so với pixel-based DM

##### Có thể train trên single GPU

##### Inference: single network pass qua decoder

#### So sánh với các phương pháp khác

##### vs VQGAN+Transformer: scales better, không cần billions of params

##### vs ADM: similar quality, 4× less compute

##### vs SR3: competitive FID, faster sampling

#### Đóng góp chính

##### Separation of compression and generative learning

##### Cross-attention conditioning mechanism

##### Democratizing high-resolution image synthesis

##### Open-source pretrained models

#### Hạn chế

##### Sequential sampling vẫn chậm hơn GANs

##### Reconstruction bottleneck cho fine-grained tasks

### Mối quan hệ giữa các bài báo

#### DDPM → DDIM

##### DDIM mở rộng DDPM với non-Markovian processes

##### Dùng cùng training objective

##### Accelerated sampling

#### DDPM/DDIM → LDM

##### LDM áp dụng diffusion trong latent space

##### Giữ nguyên core diffusion formulation

##### Thêm conditioning mechanisms

#### Timeline phát triển

##### 2020: DDPM chứng minh diffusion có thể sinh ảnh chất lượng cao

##### 2021: DDIM giải quyết vấn đề sampling chậm

##### 2022: LDM giảm computational cost, enable nhiều ứng dụng

### Các khái niệm toán học chung

#### Forward Process

##### q(xt|xt-1): thêm noise

##### β schedule: linear từ β1=10^-4 đến βT=0.02

#### Reverse Process

##### pθ(xt-1|xt): học khử noise

##### Parameterized bởi neural network εθ

#### Training Objective

##### Variational lower bound

##### Simplified MSE loss giữa predicted và actual noise

#### Sampling

##### Iterative denoising từ xT → x0

##### DDPM: stochastic

##### DDIM: deterministic hoặc stochastic (σ tunable)

## Architectural Enhancements

### U-Net Backbone from PixelCNN++

### Group Normalization

### Self-attention at Resolution 16x16

### Time Embedding: Transformer Sinusoidal Position

## Training and Sampling

### Accelerated Sampling

### Down-weighting Loss at Early Time Steps

### Use Pretrained Models for Sampling

## Performance and Benchmarks

### CIFAR10: Image Quality and Inception Score

### LSUN 256x256: Quality Comparisons

### Progressive Lossy Compression Analysis

## Theoretical Foundations

### Non-Markovian Forward Process

### Generative Process

### Shared Training Objective

## Model Variants and Extensions

### DDIM: Deterministic Diffusion Implicit Models

### LDM: Latent Diffusion Models

# Đề tài

## Mục tiêu

###  

## Kiến trúc tổng thể đề tài to

## Mục tiêu đề tài nhỏ

##  

# Questions cụ thể

## Câu hỏi cho Đề tài AI - Image Segmentation

### A. Câu hỏi chung về nghiên cứu

#### 1. Mục tiêu & Phạm vi

##### Vấn đề cốt lõi cần giải quyết là gì?

###### Segmetation cho ảnh vệ tinh

###### Cụ thể trên dataset Open earth map

##### Mục tiêu tổng quát của đề tài là gì?

###### Cải thiện performance cho dataset OEM  

###### mIoU

##### Challenges

###### Class khác nhau nhưng đặc điểm tựa tựa nhau

####### Vùng đất khó phân biệt bởi boundary

###### Tồn tại rất nhiều class trong 1 ảnh

####### -> Nhiễu

####### PP truyền thống thường ít class

######## Chưa có Model nào cho dataset OEM này

######### Cần kiểm tra lại?

##### Các chỉ số đánh giá

###### IoU từng class

###### mIoU

##### Các mục tiêu cụ thể (objectives) cần đạt được?

##### Phạm vi nghiên cứu đến đâu?

##### Những gì không nằm trong phạm vi?

##### Kết quả kỳ vọng (deliverables) là gì?

##### Có giới hạn về thời gian, ngân sách, nguồn lực không?

#### 2. Phương pháp & Đánh giá

##### Sử dụng phương pháp nghiên cứu nào?

##### Làm sao biết nghiên cứu đã thành công?

##### So sánh với nghiên cứu/giải pháp hiện có như thế nào?

#### 3. Tài nguyên & Timeline

##### Cần những tài liệu, dữ liệu, công cụ gì?

##### Ai là stakeholders?

###### Linh, Thiện

####### Giải quyết vấn đề gì

######## Semi supervised

####### Các bước làm

####### Kết quả

######## Model hoàn chỉnh

###### Vương, Vỹ

####### Giải quyết vấn đề gì

######## Thiết kế bộ denoiser

######### Các phương án

########## PA1

########### AutoEncoder

############ input (RGB image, pseudo labels)
- concat lại
- pseudo labels từ ảnh 

############ output (RGB image, pseudo labels)

########## PA2

########### Diffusion Model

############ RGB image được đưa vào condition

######### Tại sao

########## Ảnh bị noise thấp

########### Nhiễu bên latent diffusion

########## Ảnh sạch sẽ

########### đã được bỏ noise rồi

####### Các bước làm

###### Huân

####### Giải quyết vấn đề gì

####### Các bước làm

###### Danh

####### Giải quyết vấn đề gì

####### Các bước làm

##### Deadline cuối cùng là khi nào?

##### Các mốc quan trọng (milestones) là gì?

#### 4. Rủi ro

##### Những rủi ro tiềm ẩn là gì?

##### Các khó khăn có thể gặp phải?

### B. Câu hỏi đặc thù cho Image Segmentation

#### 1. Bài toán & Loại Segmentation

##### Loại segmentation nào? (Semantic/Instance/Panoptic)

###### Semantic

##### Dữ liệu ảnh thuộc lĩnh vực gì?

##### Bài toán cụ thể là gì?

##### Đây là nghiên cứu ứng dụng hay đề xuất phương pháp mới?

#### 2. Dataset

##### Sử dụng dataset công khai nào?

###### Có - OpenEarthMap (open-source) 

##### Có bao nhiêu classes/labels cần phân loại?

###### 9 classes + 1 unknown/background

##### Kích thước dataset?

###### 5000 ảnh từ 97 khu vực thuộc 44 quốc gia

##### Dữ liệu đã được annotated chưa?

###### Annotation bằng tay

####### để train

###### Annotation bằng model

####### valuation/ test [có thể]

##### Có vấn đề imbalanced data không?

###### Nhiều file trong dataset có label nhưng k chuẩn xác

####### Nhiều label không tin cậy cao

####### VD trong 5000 ảnh, có 300 ảnh label không có tin cậy cao

######## -> Chỉ lấy ảnh có tin cậy cao đem đi train

########  

##### Định dạng ảnh & kích thước ảnh?

###### RGB, 1024×1024 pixels, GeoTIFF

######  

#### 3. Model & Kiến trúc

##### Dùng model nào làm baseline?

##### Có đề xuất cải tiến kiến trúc không?

##### Pretrained backbone nào?

##### Có yêu cầu real-time inference không?

##### Yêu cầu về model size/complexity?

#### 4. Training

##### Loss function nào?

##### Optimizer & learning rate strategy?

##### Data augmentation techniques?

##### Hardware có sẵn?

##### Batch size & số epochs dự kiến?

##### Có sử dụng transfer learning không?

#### 5. Evaluation

##### Metrics đánh giá chính?

##### So sánh với những phương pháp SOTA nào?

##### Có thực hiện ablation study không?

##### Đánh giá trên test set nào?

#### 6. Deliverables

##### Có cần demo application không?

##### Format báo cáo?

##### Có yêu cầu open-source code không?

##### Ngôn ngữ viết báo cáo?

##### Có cần publication không?