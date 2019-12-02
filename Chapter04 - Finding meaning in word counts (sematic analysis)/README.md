## 4 Phân tích ngữ nghĩa - sematic analysis

Chương này bao gồm các kiến thức chính :
* Phân tích ngữ nghĩa để tạo các vector chủ đề.
* Tìm kiếm ngữ nghĩa dựa trên sự tương đồng giữa các vector chủ đề.
* Khả năng mở rộng phân tích ngữ nghĩa và tìm kiếm ngữ nghĩa với văn thể lớn.
* Sử dụng các thành phần ngữ nghĩa (các chủ đề) như là thuộc tính trong NLP pipelines
* Khảo sát không gian vector nhiều chiều

### Khái quát và giới thiệu :
Đối với kỹ thuật TF-IDF (term frequency-inverse document frequency vectors) giúp ta ước lượng được tầm quan trọng của từ trong 1 tập hợp văn bản.

Những bài học NLP trước đây đã giới thiệu về 1 thuật toán nhằm mở ra ý nghĩa của từ kết hợp và các vector (tính toán) để thể hiện ý nghĩa này. Thuật toán này là LSA (latent semantic analysis). Và khi bạn sử dụng công cụ này, không chỉ bạn có thể thể hiện ý nghĩa của từ thông qua các vector, mà bạn có thể sử dụng chúng để thể hiện ý nghĩa của cả văn bản.

Trong chương này, bạn sẽ được học về những ngữ nghĩa hay là các vector chủ để (topic vectors). Bạn sẽ sử dụng các trọng số frequency scores từ TF-IDF vectors để tính toán các topic scores mà tạo nên các chiều của topic vector.

 Những topic vectors này có thể giúp bạn làm những điều hết sức thú vị, chúng có thể được sử dụng để tìm kiếm các văn bản dựa trên nghĩa của nó - sematic search. Hầu hết, semantic search trả về kết quả tìm kiếm tốt hơn nhiều so với việc tìm kiếm keyword (keyword search) - hay còn đc implement bởi TF-IDF search. 

Ngoài ra, bạn có thể sử dụng những vector ngữ nghĩa (semantic vectors) để xác định các từ và n-grams mà tốt nhất để thể hiện các chủ đề của tài liệu, tập hợp văn bản. Và với những vector này và mối liên hệ giữa chúng, bạn có thể cung cấp cho ai đó những từ có thông tin có ý nghĩa nhất cho một văn bản - một tập các keywords mà tóm tắt ý nghĩa của nó. 

Và bạn có thể so sánh bất kỳ 2 văn bản và cho ra sự gần nghĩa giữa chúng.

***TIP:*** Cái khái niệm chủ đề (topic), ngữ nghĩa (semantic) và ý nghĩa (meaning) đều có nghĩa giống nhau và thường được sử dụng hoán đổi cho nhau khi nói về NLP. Trong chương này, bạn sẽ được học các xây dựng một NLP pipeline mà có thể tìm sự tương đồng về ngữ nghĩa của 3 từ này này. NLP pipeline của bạn thậm chí có thể tìm ra sự tương đồng về ngữ nghĩa của các cụm từ "figure out" và từ "compute". Máy thì chỉ có thể "compute" meaning, chứ không "figure out" meaning. 



### 4.1 Từ word counts đến topic scores

Bạn đã biết như thế nào để đếm sự lặp lại (tần số - frequency) của các từ. Và biết làm như thế nào để ghi lại sự quan trọng của từ trong TF-IDF vector hay ma trận. Nhưng như vậy vẫn chưa đủ, bạn muốn ghi lại ý nghĩa, chủ đề mà các từ đó được sử dụng cho.

#### 4.1.1 TF-IDF và Lemmatization

Khái niệm : Lemmatization hay Stemming : tạo một bộ các từ có cùng cách phát âm (các chữ giống nhau. *VD : build, builder, building, . . .*), và thường là sẽ ngữ nghĩa giống nhau. Sau đó dán nhãn mới cho các tập lemma này và sử dụng tokens với nhãn này thay các từ nguyên bản tạo ra các lemmatized TF-IDF.
**Chú ý :** Cách tiếp cận lemmatization này gộp cho những từ có cách phát âm giống nhau trong việc thống kê, nhưng các từ này không cần phải cùng nghĩa (khái niệm từ đồng nghĩa khá phức tạp). Vì vậy kết của sau cùng, cho ta 2 tập các từ (chunks of text) mà cùng nói về 1 thứ, nhưng dùng những từ không gần với các từ trong lại trong mô hình không gian lemmatized TF-IDF vector.

#### 4.1.2 Vectors chủ đề (Topic vectors)
Ý nghĩa : bạn cần có một cách để trích xuất nhưng thông tin, ý nghĩa thêm từ những vectors thống kê (TF-IDF), cần một sự ước lượng tốt hơn về dấu hiệu gì những từ trong văn bản đang nắm giữ, và bạn cần biết ý nghĩa của kết hợp các từ trong một văn bản cụ thể. Bạn muốn biểu diện ý nghĩa đó vào một vectors như TF-IDF nhưng compact hơn (đầy đủ) và nhiều ý nghĩa hơn.

Và ta gọi những vector compact (tinh gọn) đó là vector "word-topic-vectors", gọi  vector ý nghĩa của văn bản là "document-topic-vectors", và ta có thể gọi cả 2 loại vector này là các "topic vectors", miễn là bạn phân biệt được rõ ràng rằng topic vectors này là cho từ hay cho văn bản.

Những topic vectors này có thể compact hay được mở rộng (nhiều chiều) như ý bạn. LSA topic vectors có thể có thể có ít chiều như 1 chiều hay có thể lên tới hàng ngàn chiều.

Bạn có thể cộng, trừ các topic vectors trong chapter 4 này như những vector khác, nhưng lần này thì tổng và hiệu có ý nghĩa nhiều hơn trong TF-IDF vectors(chapter 3). Và quan trọng là khoảng cách giữa các topic vectors hữu dụng cho những việc như phân nhóm (clustering) các văn bản hay tìm kiếm ngữ nghĩa (semantic search). Trước đó, bạn có thể phân đoạn và tìm kiếm sử dụng keywords và TF-IDF vectors, bây giờ, bạn có 1 công cụ nữa để làm công việc này sử dụng ngữ nghĩa, meaning !

Khi hoàn thành, bạn sẽ có một document-topic vectors cho mỗi document trong tập văn bản. Và quan trọng hơn nữa, đó là bạn sẽ không phải tái xử lý lại toàn bộ tập văn bản để tính toán topic vector mới cho văn bản mới đó. Bạn sẽ có một topic vector cho mỗi từ trong tập từ vựng, và bạn có thể sử dụng các word topic vectors để tính toán topic vector cho mọi văn bản đã sử dụng những từ đó.

**TIP:** Một số thuật toán để tạo topic vectors như Latent Dirichlet Allocation (LDA), cần yêu cầu bạn phải tái xử lý toàn bộ tập văn bản, mỗi khi bạn thêm vào 1 văn bản mới.

#### 4.1.3. Thought experiment
Thử đọc và chạy code VD để hiểu hơn về implement, ở đây giả sử là bạn có một cố TF-IDF vecor cho một văn bản cụ thể, và bạn muốn chuyển đổi nó thành topic vector. Hãy nghĩ xem là liệu từ này đóng góp cho topic như thế nào. Ví dụ trong file [example_chapter4.py](https://github.com/nducthang/NCKH_2020/blob/master/Chapter04%20-%20Finding%20meaning%20in%20word%20counts%20(sematic%20analysis)/example_chapter4.py) (dòng 1 - 51).

**Chú ý : ** Chúng ta chọn trọng số có dấu của từ để tạo ra các topic vectors. Điều này cho phép bạn có thể sử dụng các trọng số ấm cho từ có nghĩa đối ngược với topic. Và trong ví dụ này ta làm bằng tay, ta chọn chuẩn cho topic vector theo chuẩn L1-norm (chuẩn Manhattan). Tuy nhiên, mô hình thực sự LSA mà sẽ đề cập tiếp trong Chapter này chuẩn hóa topic vectors bởi chuẩn sử dụng rộng rãi hơn là chuẩn L2-norm (conventional Euclidean distance)

**Chú ý :** Trong toán học, kích thước của tập từ vựng (vocabulary) thường được kí hiệu là |V|. Và biến V một mình dùng để chỉ tập của các từ có thể thuộc tập từ vựng. Nên, nếu bạn đang định viết một bài báo học thuật về NLP, chú ý đến kí hiệu |V| mỗi khi muốn đến cập đến kích cỡ của tập từ vựng.

#### 4.1.4 Một thuật toán để tính toán topic vectors
> You shall know a word by the company it keeps - J.R.Firth.

LSA là một thuật toán để phân tích ma trận TF-IDF (bảng các vector TF-IDF) để tập hợp các từ lại thành các topics. Nó có thể hoạt động trên BOW, nhưng bằng thực nghiệm đã cho thấy thực thi thuật toán trên TF-IDF cho kết quả tốt hơn đáng kể.

LSA cũng tối ưu những topics để duy trì sự đa dạng trong số chiều topics; khi bạn dùng những topics mới này thay vì sử dụng từ gốc, bạn vẫn có thể ghi lại những ý nghĩa (ngữ nghĩa) của văn bản. Số topics bạn cần cho mô hình để ghi lại ngữ nghĩa của văn bản là nhỏ hơn rất nhiều (far less than) so với số từ trong bộ từ vựng của các vectors TF-IDF. Cho nên LSA thường được gọi là một thuật toán để giảm số chiềum hay là một kỹ thuật để giảm số chiều. LSA giảm số chiều mà bạn cần đề có thể quan sát ý nghĩa của văn bản bạn đang xét đến.

**Tóm tắt:** Chắc hẳn bạn đã đọc về những ký thuật giảm số chiều cho ma trận lớn, hay trong các bài toán về xử lý ảnh với độ phân giải cao (high dimensions data). Có thể bạn đã đụng đến kỹ thuật có tên là PCA (principal component analysis). Hiểu rặng, kỹ thuật về mặt toán của PCA là giống với LSA, tuy nhiên , khi nói đến PCA ta hiều rằng bạn đang cố giảm số chiều của 1 tấm ảnh hay của một ma trận, thay vì giảm số chiều của các ma trận BOW hay TF-IDF.

**Chú ý:** Đánh chỉ số là việc cơ sở dữ liệu (database) có thể truy ra một hàng cụ thể một cách nhanh chóng dựa trên một số thông tin khác nhau bạn cung cấp về cột đó. Một đánh chỉ số sổ (textbook's index) cũng hoạt động tương tự như vậy. Nếu bạn đang tìm kiếm một trang cụ thể, bạn có thể tìm kiếm từ trong chỉ số mà có lẽ trong trang đó, sau đó bạn có thể đến thằng trang đó mà chứa từ đó.

* **LSA "COUSINS"**
Có hai thuật toán giống với LSA, với cùng những ứng dụng trong NLP, cho nên trong chương này sẽ đề cập đến đó là :
	1. Linear Discriminant analysis (LDA)
	2. Latent Dirichlet allocation (LDiA)

LDA phá vỡ một văn bản thành chỉ 1 topic. Còn LDiA thì giống LSA hơn bởi vì nó thể phá một văn bản thành số lượng topic tùy ý.

**TIP:** Bởi vì chỉ có 1 chiều, nên LDA không cần yêu cấu SVD (singular value decomposition). Bạn có thể chỉ tính toán centroid (trung bình hay kỳ vọng) của tất cả các vector TF-IDF cho mỗi vế của lớp nhị phân (như 0 - 1). Chiều sau đó trở thành tuyến tính (1 đường thằng) giữa 2 centroids. Sau đó, một vector TF-IDF thuộc về đường thằng đó ( dot product của vector TF-IDF với đường đó) cho ta biết sự gần với lớp này hay lớp kia. Phần còn lại của chapter là nói về implementation của các thuật toán LDA, LSA, LDiA theo thứ tự, áp dụng vào bài toán nhận biết tin nhắn SPAM.

Các cài đặt đều rất dễ hiểu, nhưng đối với dữ liệu SMS - Spam, các mô hình thô này cho ta một kết quả có độ chính xác rất bất ngờ. Các phần tiếp theo chủ yếu là cài đặt, và các nhận xét của tác giả, ta có thể đọc lướt nhanh trong tài liệu và đồng thời thực hiện code luôn. Trong tài liệu chỉ giới thiệu cách implementation chứ không đi xâu vào nguồn gốc toán học của các thuật toán. Nếu bạn không chỉ muốn làm deployment thì cần phải đọc về các thuật toán như LSA (kém với SVD, PCA) hay LDiA, LDA trong các tài liệu khác, còn nếu chỉ làm về ứng dụng, thì có thể đọc các nhận xét khi cài đặt trong các phần trong cuốn sách này. Các chú ý hay nhận xét nếu viết ra không đi cùng với cài đặt thì rất vô nghĩa, nên trong lúc cài đặt cùng có những comment line cần được chú ý. Tất cả phần cái đặt của các classifier trong các phần trong chapter được cài tại file file [example_chapter4.py](https://github.com/nducthang/NCKH_2020/blob/master/Chapter04%20-%20Finding%20meaning%20in%20word%20counts%20(sematic%20analysis)/example_chapter4.py) (dòng 51 - end).


#### 4.1.5 Mô hình LDA
**Tóm Tắt:** Mô hình training chỉ gồm 3 bước sau :
	1. Tính toán các vị trí trung bình (centroid) của tất cả các vectors TF-IDF trong một lớp (ở đây có 2 lớp, 0-1 ứng với NonSPAM-SPAM).
	2. ính toán các vị trí trung bình (centroid) của tất cả các vectors TF-IDF trong lớp còn lại.
	3. Tính toán vector khác nhau (vector difference) giữa các centroids(đường thằng mà kết nối giữa chúng)

**Nhận xét:** LDA là supervised algorithm, mục tiêu là đi tìm vector (đường thằng) giữa 2 centroids của lớp nhị phân, để dự đoán với mô hình này, bạn cần phải tìm ra rằng : một vector TF-IDF mới gần hơn số với centroid của một lớp (SPAM) này hơn gần hơn số với centroid của lớp (NonSPAM) kia.


### 4.2 LSA (Latent Semantic Analysis)
### 4.3 SVD (Singular Value Decomposition)
### 4.4 PCA (Principal Component Analysis)
### 4.5 LDia (Latent Dirichlet Allocation)
### 4.6 Khoảng cách và sự giống nhau (Distance and Similarity)
Ta cần phải xem lại những điểm giống nhau (similarity scores) đã nói trong chapter 2 và chapter 3 để chắc chắn rằng không gian vector là hoạt động với chúng. Nhớ rằng , bạn có thể dùng điểm giống nhau (hay khoảng cách) để chỉ ra sự giống hay xa nhau giữa 2 văn bản dựa trên sự giống nhau (hay khoảng cách) của các vector bạn đã sử dụng để biểu diễn chúng.

Bạn có thể dùng những điểm giống nhau (similarity scores hoặc k/c) để thấy được mô hình LDA hoạt động tốt như thế nào với mô hình TF-IDF nhiều chiều trong chapter 3.

Khoảng cách giữa các vectors đặc trưng (feature vectors) như word vectors, topic vectors, document context vectors, etc. ảnh hưởng đến hiệu năng của một NLP pipeline hoặc mọi machine learning pipeline khác. Vậy lựa chọn các tính khoảng cách nào để bạn có thể sử dụng trong không gian nhiều ? Và chọn cái nào cho một vấn đề NLP cụ thể? Một số cách dùng phổ biến trong nhiều ví dụ có thể thân thuộc từ hình học hay đại số tuyển tính, nhưng một số có thể là mới với bạn :

	- Euclidean hay Cartesion distance hay Root Mean Square Error (RMSE)
	- Squared Euclidean distance, sum of squares distance (SSD) : L22
	- Cosin hay angular hay projected distance : normalized dot product
	- Minkowski distance: p-norm hay Lp
	- Fractional distance, frantional norm: p-norm hay Lp với 0 < p < 1
	- City block, Manhattan hay taxicab distance, sum of absolute distance(SAD) : l-norm hay L1
	- Jaccard distance, inverse set similarity
	- Mahalonobis distance
	- Levenshtein hay edit distance

Sự đa dạng trong việc tính toán khoảng cách như là một "di chúc" thể hiện rằng tầm quan trọng của nó là như thế nào.

**Chú ý:** Các khoảng cách này đều có thể tính toán trên python qua Scikit-learn.

**Chú ý:** Khái niệm khoảng cách (distance) hay độ dài (length) thường bị nhầm lẫn với khái niệm metric (tạm dịch là số liệu, nên để dịch gốc metric), bởi vì nhiều khoảng cách được dùng là metric, nhưng không phải tất cả. Một điều gây nhầm lẫn nữa là metric cũng có thể đôi khi được gọi là "distance functions" (hàm khoảng cách) hay "distance metrics" trong toán học hàn lâm và các định lý. Metrics là khái niệm toán học được định nghĩa trên 4 tiên đề (Nonnegativity-Không âm, Indiscernibility-Không phân biệt, Symmertry-Đối xứng, Triangle equality-Bắc cầu) Một khái niệm cũng khá nhập nhằng là khái niệm "measure" (tạm dịch độ đo là khái niệm toán học liên quan đến kích thước của một tập các đối tượng), cho nên từ "measure" nên được dùng cẩn thận để miêu tả bất kỳ chỉ số hay thống kê nào suy ra từ một đối tượng hay kết hợp của các đối tượng trong NLP.

### 4.7 Định hướng và phản hồi (Steering with feedback)
### 4.8 Khả năng của Topic vector 
Hai phần 4.7, 4.8 nói về những đánh giá chung về phương pháp sử dụng topic vectos.

./