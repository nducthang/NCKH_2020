## 4. Phân tích ngữ nghĩa - sematic analysis


**Chương này bao gồm các kiến thức chính :**
* Phân tích ngữ nghĩa để tạo các vector chủ đề
* Tìm kiếm ngữ nghĩa dựa trên sự tương đồng giữa các vector chủ đề
* Khả năng mở rộng phân tích ngữ nghĩa và tìm kiếm ngữ nghĩa với văn thể lớn
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




### 4.1. Từ word counts đến topic scores

Bạn đã biết như thế nào để đếm sự lặp lại (tần số - frequency) của các từ. Và biết làm như thế nào để ghi lại sự quan trọng của từ trong TF-IDF vector hay ma trận. Nhưng như vậy vẫn chưa đủ, bạn muốn ghi lại ý nghĩa, chủ đề mà các từ đó được sử dụng cho.

#### 4.1.1. TF-IDF và Lemmatization

Khái niệm : Lemmatization hay Stemming : tạo một bộ các từ có cùng cách phát âm (các chữ giống nhau. *VD : build, builder, building, . . .*), và thường là sẽ ngữ nghĩa giống nhau. Sau đó dán nhãn mới cho các tập lemma này và sử dụng tokens với nhãn này thay các từ nguyên bản tạo ra các lemmatized TF-IDF.
**Chú ý :** Cách tiếp cận lemmatization này gộp cho những từ có cách phát âm giống nhau trong việc thống kê, nhưng các từ này không cần phải cùng nghĩa (khái niệm từ đồng nghĩa khá phức tạp). Vì vậy kết của sau cùng, cho ta 2 tập các từ (chunks of text) mà cùng nói về 1 thứ, nhưng dùng những từ không gần với các từ trong lại trong mô hình không gian lemmatized TF-IDF vector.

#### 4.1.2. Vectors chủ đề (Topic vectors)
Ý nghĩa : bạn cần có một cách để trích xuất nhưng thông tin, ý nghĩa thêm từ những vectors thống kê (TF-IDF), cần một sự ước lượng tốt hơn về dấu hiệu gì những từ trong văn bản đang nắm giữ, và bạn cần biết ý nghĩa của kết hợp các từ trong một văn bản cụ thể. Bạn muốn biểu diện ý nghĩa đó vào một vectors như TF-IDF nhưng compact hơn (đầy đủ) và nhiều ý nghĩa hơn.

Và ta gọi những vector compact (tinh gọn) đó là vector "word-topic-vectors", gọi  vector ý nghĩa của văn bản là "document-topic-vectors", và ta có thể gọi cả 2 loại vector này là các "topic vectors", miễn là bạn phân biệt được rõ ràng rằng topic vectors này là cho từ hay cho văn bản.

Những topic vectors này có thể compact hay được mở rộng (nhiều chiều) như ý bạn. LSA topic vectors có thể có thể có ít chiều như 1 chiều hay có thể lên tới hàng ngàn chiều.

Bạn có thể cộng, trừ các topic vectors trong chapter 4 này như những vector khác, nhưng lần này thì tổng và hiệu có ý nghĩa nhiều hơn trong TF-IDF vectors(chapter 3). Và quan trọng là khoảng cách giữa các topic vectors hữu dụng cho những việc như phân nhóm (clustering) các văn bản hay tìm kiếm ngữ nghĩa (semantic search). Trước đó, bạn có thể phân đoạn và tìm kiếm sử dụng keywords và TF-IDF vectors, bây giờ, bạn có 1 công cụ nữa để làm công việc này sử dụng ngữ nghĩa, meaning !

Khi hoàn thành, bạn sẽ có một document-topic vectors cho mỗi document trong tập văn bản. Và quan trọng hơn nữa, đó là bạn sẽ không phải tái xử lý lại toàn bộ tập văn bản để tính toán topic vector mới cho văn bản mới đó. Bạn sẽ có một topic vector cho mỗi từ trong tập từ vựng, và bạn có thể sử dụng các word topic vectors để tính toán topic vector cho mọi văn bản đã sử dụng những từ đó.

**TIP:** Một số thuật toán để tạo topic vectors như Latent Dirichlet Allocation (LDA), cần yêu cầu bạn phải tái xử lý toàn bộ tập văn bản, mỗi khi bạn thêm vào 1 văn bản mới.

#### 4.1.3. Chạy thử nghiệm
Thử đọc và chạy code VD để hiểu hơn về implement, ở đây giả sử là bạn có một cố TF-IDF vecor cho một văn bản cụ thể, và bạn muốn chuyển đổi nó thành topic vector. Hãy nghĩ xem là liệu từ này đóng góp cho topic như thế nào. Ví dụ trong file [example_chapter4.py](https://github.com/nducthang/NCKH_2020/blob/master/Chapter04%20-%20Finding%20meaning%20in%20word%20counts%20(sematic%20analysis)/example_chapter4.py) (dòng 1 - 51).

*Chú ý:* Chúng ta chọn trọng số có dấu của từ để tạo ra các topic vectors. Điều này cho phép bạn có thể sử dụng các trọng số ấm cho từ có nghĩa đối ngược với topic. Và trong ví dụ này ta làm bằng tay, ta chọn chuẩn cho topic vector theo chuẩn L1-norm (chuẩn Manhattan). Tuy nhiên, mô hình thực sự LSA mà sẽ đề cập tiếp trong Chapter này chuẩn hóa topic vectors bởi chuẩn sử dụng rộng rãi hơn là chuẩn L2-norm (conventional Euclidean distance).

**Chú ý :** Trong toán học, kích thước của tập từ vựng (vocabulary) thường được kí hiệu là |V|. Và biến V một mình dùng để chỉ tập của các từ có thể thuộc tập từ vựng. Nên, nếu bạn đang định viết một bài báo học thuật về NLP, chú ý đến kí hiệu |V| mỗi khi muốn đến cập đến kích cỡ của tập từ vựng.

#### 4.1.4. Một thuật toán để tính toán topic vectors
> You shall know a word by the company it keeps - J.R.Firth.

LSA là một thuật toán để phân tích ma trận TF-IDF (bảng các vector TF-IDF) để tập hợp các từ lại thành các topics. Nó có thể hoạt động trên BOW, nhưng bằng thực nghiệm đã cho thấy thực thi thuật toán trên TF-IDF cho kết quả tốt hơn đáng kể.

LSA cũng tối ưu những topics để duy trì sự đa dạng trong số chiều topics; khi bạn dùng những topics mới này thay vì sử dụng từ gốc, bạn vẫn có thể ghi lại những ý nghĩa (ngữ nghĩa) của văn bản. Số topics bạn cần cho mô hình để ghi lại ngữ nghĩa của văn bản là nhỏ hơn rất nhiều (far less than) so với số từ trong bộ từ vựng của các vectors TF-IDF. Cho nên LSA thường được gọi là một thuật toán để giảm số chiềum hay là một kỹ thuật để giảm số chiều. LSA giảm số chiều mà bạn cần đề có thể quan sát ý nghĩa của văn bản bạn đang xét đến.

**Tóm tắt:** Chắc hẳn bạn đã đọc về những ký thuật giảm số chiều cho ma trận lớn, hay trong các bài toán về xử lý ảnh với độ phân giải cao (high dimensions data). Có thể bạn đã đụng đến kỹ thuật có tên là PCA (principal component analysis). Hiểu rặng, kỹ thuật về mặt toán của PCA là giống với LSA, tuy nhiên , khi nói đến PCA ta hiều rằng bạn đang cố giảm số chiều của 1 tấm ảnh hay của một ma trận, thay vì giảm số chiều của các ma trận BOW hay TF-IDF.

**Chú ý:** Đánh chỉ số là việc cơ sở dữ liệu (database) có thể truy ra một hàng cụ thể một cách nhanh chóng dựa trên một số thông tin khác nhau bạn cung cấp về cột đó. Một đánh chỉ số sổ (textbook's index) cũng hoạt động tương tự như vậy. Nếu bạn đang tìm kiếm một trang cụ thể, bạn có thể tìm kiếm từ trong chỉ số mà có lẽ trong trang đó, sau đó bạn có thể đến thằng trang đó mà chứa từ đó.

 **LSA "COUSINS" (Các thuật toán "sinh đôi" với LSA) **
Có hai thuật toán giống với LSA, với cùng những ứng dụng trong NLP, cho nên trong chương này sẽ đề cập đến đó là :

	1. Linear Discriminant analysis (LDA)

	2. Latent Dirichlet allocation (LDiA)

LDA chia một văn bản thành chỉ 1 topic. Còn LDiA thì giống LSA hơn bởi vì nó thể chia một văn bản thành số lượng topic tùy ý.

**TIP:** Bởi vì chỉ có 1 chiều, nên LDA không cần yêu cấu SVD (singular value decomposition). Bạn có thể chỉ tính toán centroid (trung bình hay kỳ vọng) của tất cả các vector TF-IDF cho mỗi vế của lớp nhị phân (như 0 - 1). Chiều sau đó trở thành tuyến tính (1 đường thằng) giữa 2 centroids. Sau đó, một vector TF-IDF thuộc về đường thằng đó ( dot product của vector TF-IDF với đường đó) cho ta biết sự gần với lớp này hay lớp kia. Phần còn lại của chapter là nói về implementation của các thuật toán LDA, LSA, LDiA theo thứ tự, áp dụng vào bài toán nhận biết tin nhắn SPAM.

Các cài đặt đều rất dễ hiểu, nhưng đối với dữ liệu SMS - Spam, các mô hình thô này cho ta một kết quả có độ chính xác rất bất ngờ. Các phần tiếp theo chủ yếu là cài đặt, và các nhận xét của tác giả, ta có thể đọc lướt nhanh trong tài liệu và đồng thời thực hiện code luôn. Trong tài liệu chỉ giới thiệu cách implementation chứ không đi xâu vào nguồn gốc toán học của các thuật toán. Nếu bạn không chỉ muốn làm deployment thì cần phải đọc về các thuật toán như LSA (kém với SVD, PCA) hay LDiA, LDA trong các tài liệu khác, còn nếu chỉ làm về ứng dụng, thì có thể đọc các nhận xét khi cài đặt trong các phần trong cuốn sách này. Các chú ý hay nhận xét nếu viết ra không đi cùng với cài đặt thì rất vô nghĩa, nên trong lúc cài đặt cùng có những comment line cần được chú ý. Tất cả phần cái đặt của các classifier trong các phần trong chapter được cài tại file [example_chapter4.py](https://github.com/nducthang/NCKH_2020/blob/master/Chapter04%20-%20Finding%20meaning%20in%20word%20counts%20(sematic%20analysis)/example_chapter4.py) (dòng 51 - end).


#### 4.1.5. Mô hình LDA
**Tóm Tắt:** Mô hình training chỉ gồm 3 bước sau :

	1. Tính toán các vị trí trung bình (centroid) của tất cả các vectors TF-IDF trong một lớp (ở đây có 2 lớp, 0-1 ứng với NonSPAM-SPAM).

	2. Tính toán các vị trí trung bình (centroid) của tất cả các vectors TF-IDF trong lớp còn lại.

	3. Tính toán vector khác nhau (vector difference) giữa các centroids(đường thằng mà kết nối giữa chúng)

**Nhận xét:** LDA là supervised algorithm, mục tiêu là đi tìm vector (đường thằng) giữa 2 centroids của lớp nhị phân, để dự đoán với mô hình này, bạn cần phải tìm ra rằng : một vector TF-IDF mới gần hơn số với centroid của một lớp (SPAM) này hơn gần hơn số với centroid của lớp (NonSPAM) kia.



### 4.2. LSA (Latent Semantic Analysis)

LSA (tạm dịch : phân tích ngữ nghĩa ngầm) được dựa theo kĩ thuật lâu đời và được sử dụng nhiều nhất để giảm số chiều, SVD. Thậm chí SVD đã được sử dụng rộng rãi trước khi Machine Learning ra đời. SVD phân tích một ma trận thành 3 ma trận, một trong số đó là ma trận chéo.

Sử dụng SVD, LSA có thể phân ma trận TF-IDF thành 3 ma trận đơn giản hơn. Và chúng có thể được nhân lại với nhau để tạo thành ma trận ban đầu, mà không bị thay đổi. Bạn có thể cắt bớt những ma trận đó (loại bỏ 1 số hàng hoặc cột) trước khi nhân chúng lại với nhau, điều này làm giảm số chiều của mà bạn phải sử dụng trong mô hình không gian vector.

Những ma trận bị cắt bớt không đưa ra chính xác ma trận TF-IDF ban đầu, chúng đưa cho bận 1 điều tốt hơn. Sự biểu diễn mới này của các văn bản chứa bản chất, "ngữ nghĩa ngầm" của các văn bản đó. Đó là lí do vì sao SVD được sử dụng ở nhiều lĩnh vực khác như lĩnh vực nén (compression). Nó ghi lại bản chất của tập dữ liệu và loại bỏ nhiễu. Một file ảnh JPEG có kích thước nhỏ hơn đến 10 lần so với bản nguyên gốc bitmap, nhưng nó vẫn chứa tất cả các thông tin của ảnh nguyên mẫu.

Khi bạn sử dụng SVD trong NLP, bạn gọi nó là LSA, LSA khám phá ra ngữ nghĩa, ý nghĩa của các từ được giấu và đang đợi để được khám phá. LSA là một công cụ toán học để tìm các tốt nhất để biến đổi tuyến tính : quay và nới rộng (linearly transform : rotate and stretch) mọi tập các vector NLP, như các vector TF-IDF hay là các vector BOW. Và cách tốt nhất cho nhiều ứng dụng là xếp các trục tọa độ (dimensions) trong vector mới với độ bao phủ (spread) tốt nhất hay là phương sai (variance) trong tần suất của từ là tốt nhất (word frequencies). Sau đó, bạn có thể loại những chiều đó trong không gian vector mới mà không đóng góp nhiều cho phương sai trong các vector từ văn bản tới văn bản. Sử dụng SVD theo cách này còn được gọi là truncated SVD.

LSA sử dụng SVD để tìm các sự kết hợp của các từ được kết hợp với nhau cho phương sai lớn nhất trong tập dữ liệu. Có thể hiểu là LSA giúp loại bỏ các chiều (topics) mà có phương sai ít nhất giữa các văn bản, do các chủ đề có lượng phương sai thấp thường là yếu tố nhiễu (noise) trong các thuật toán machine learning

#### 4.2.1. Ví dụ cài đặt
Đọc trong code kèm thêm các nhận xét

### 4.3. SVD (Singular Value Decomposition)

SVD là thuật toán đằng sau LSA
**Nhận xét:** Khi bạn chạy SVD trên ma trận BOW hay TF-IDF. SVD sẽ tìm các kết hợp của các từ mà thuộc về nhau. SVD tìm những từ cùng xuất hiện đó bằng cách tính độ tương quan (correlation) giữa các cột (terms) của ma trận (term-document matrix). SVD đồng thời tìm độ tương quan của term (cột trong ma trận) giữa các văn bản và độ tương quan của các văn bản với các văn bản khác. Với 2 thông tin này, SVD cũng tính toán tổ hợp tuyến tính của các terms mà có phương sai lớn nhất giữa các tập văn bản (corpus). Tổ hợp tuyến tính của các term-frequencies này sẽ trở thành các chủ đề.

SVD sẽ tập hợp các terms có hệ số tương quan cao với nhau (do chúng cùng xuất hiện trong văn bản thường xuyên) và cũng đa dạng qua tập văn bản. Ta coi các tổ hợp tuyến tính của các từ này như là các chủ đề (topics), những chủ đề này biến các vectors BOW hay TF-IDF thành các vector chủ đề mà sẽ cho bạn biết chủ đề mà 1 văn bản nói về. Một vector chủ đề giống như một bản tóm tắt, tổng quát về cái mà văn bản thể hiện.

**Chú ý:** SVD là một thuật toán bread-and-butter (idioms-quan trọng) trong numpy, chứ không phải trong scikit-learn như nhiều người nhầm lẫn. Để hiểu hơn về thuật toán SVD, ý nghĩa toán học (đọc thêm trong cuốn *ML cơ bản - Vũ Hữu Tiệp*).

Thuật toán SVD đằng sau LSA chú ý rằng : nếu những từ mà luôn được sử dụng cùng nhau thì cho chúng vào một topic. Đó là cách nó có thể giảm số chiều, ngay cả khi bạn không có kế hoạch sử dụng mô hình chủ đề (topic model) trong pipeline, LSA (SVD) vẫn có thể là 1 cách rất tốt để nén các ma trận từ - văn bản (word-document matrices) và xác định tiềm nằng hợp các từ hay n-grams cho pipeline.

 
### 4.4. PCA (Principal Component Analysis)
PCA (tạm dịch : phân tích thành phần chủ yếu) là một tên khác cho SVD khi mó được sử dụng để giảm số chiều, giống như bạn đã làm để hoàn thành LSA trước đó, và mô hình PCA trong scikit-learn có một số hiệu chỉnh về mặt toán học mà sẽ giúp tăng độ chính xác cho NLP pipeline của bạn.

**Chú ý:** Trong sklearn.PCA, có một số hiệu chỉnh như : nó tự động làm tròn (centers) tập dữ liệu bằng cách trừ đi kỳ vọng trung bình của tần suất từ (word frequencies) và sử dụng 1 kỹ thuật là dùng một hàm *flig_sign* để tự nhiên tính dấu của các singular vectors. Cuối cùng, trong sklearn phần cài đặt PCA, còn cho thêm một bước tùy trọn "whitening". Bước này giống với cả thủ thuật để loại bỏ các singular values khi biến đổi các word-document vectors thành các topic-document vectors (Thay vì đặt tất cả các singular values trong S về một, whitening chia dữ liệu cho phương sai, khá giống với biến đổi sklearn.StandardScaler). Thủ thuật này giúp mở rộng tập dữ liệu và giúp cho các thuật toán tối ưu khong bị kẹt trong "half pipes" hay còn gọi là "rivers" của dữ liệu mà có thể nảy sinh khi các thuộc tính trong tập dữ liệu liên quan đến nhau. 

Trong hầu hết các bài toán thực tế, bạn sẽ muốn sử dụng mô hình PCA trong sklearn.PCA cho Latent semantic analysis. Một ngoài lệ đó là khi bạn có quá nhiều văn bản mà không thể đọc được bằng RAM, khi đó, bạn cần phải sử dụng IncrementalPCA model trong sklearn hoặc sử dụng 1 số kỹ thuật scaling khác nữa.

**Tóm tắt:** Kỹ thuật LSA và SVD nâng cao. Sự thành công của SVD cho việc phân tích ngữ nghĩa cũng như giảm chiều ma trận đã thúc đẩy các nhà nghiên cứu để mở rộng nó, những mở rộng này hầu như đều dành cho các các vẫn đề không nằm trong NLP, nhưng chúng thi thoảng được sử dụng cho các hệ thống khuyến nghị (recommendation engines) dựa trên biểu hiện, và chúng đã được dùng trong NLP về thổng kê các phần của phát biểu (part of speech). Nên là, một số mô hình nâng cao sau có thể bạn sẽ cân nhắc để tìm hiểu thêm và có thể có hiệu quả hơn khi áp dụng vào mô hình phân tích ngữ nghĩa pipeline sau này : Quadratic discriminant analysis (QDA), Random projection, Nonnegative matrix factorization (NMF).


### 4.5. LDia (Latent Dirichlet Allocation)
LDiA ý tưởng : hoàn toàn giống với cách bạn sử dụng LSA (và SVD ẩn sau đó) để tạo ra mô hình chủ đề, nhưng khác là LDiA sử dụng [phân phối Dirichlete] (https://en.wikipedia.org/wiki/Dirichlet_distribution) cho tần suất từ (word frequencies). Nó chính xác hơn về thống kê của vị trị các từ trong chủ đề hơn tuyến tính về mặt toán học trong LSA (nhắc lại : LSA tạo là phép biến đổi tuyến tính có thể áp dụng cho một tập mới mà không phải training).

LDiA cho rằng mỗi văn bản là một hỗn hợp (mixture - linear combination) của một số topics bất kỳ mà bạn chọn trước khi training LDiA model. LDiA cũng giả sử rằng mỗ topic có thể được biểu diễn bởi một phân phối của các từ. Xác suất hay trọng số (weight) cho mỗi chủ đề trong một văn bản, cũng như xác suất của 1 từ được phân cho 1 chủ đề, được giả sử từ ban đầu là một phân phối Dirichlete (the prior). Chính ý tưởng cốt lõi này thể hiện trong tên của thuật toán.

**TIP:** Topic mix cho 1 văn bản có thể được xác định bởi các sự kết hợp từ trong mỗi chủ đề mà các chủ đề đó gồm các từ được phân bố. Điều này khiến LDiA topic model trở nên dễ hiểu hơn, bởi vì từ được chỉ định cho chủ đề, chủ đề chỉ định cho văn bản, tạo nên nhiều ý nghĩa (hiểu) hơn so với LSA. Cho nên trong các bài toán thực tế cần (như Business intelligent) phải giải thích cho các doanh nghiệp, mà mô hình vẫn có tính hiệu quả, nên cân nhắc tới LDiA.



### 4.6. Khoảng cách và sự giống nhau (Distance and Similarity)
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



### 4.7. Định hướng và phản hồi (Steering with feedback)



### 4.8. Khả năng của Topic vector 
Hai phần 4.7, 4.8 nói về những đánh giá về ưu điểm, nhược điểm của phương pháp sử dụng topic vector. 



### Tổng kết:

- Bạn có thể dùng SVD cho phân tích ngữ nghĩa (semantic analysis) để phân rã và biến đổi TF-IDF và BOW vectors thành các vectors chủ đề

- Dùng LDiA khi bạn cần tính toàn những vectors chủ đề có thể giải thích được.

- Không quan trọng bạn tạo ra vectors chủ để như thế nào, chúng đều có thể được sử dụng cho tìm kiếm ngữ nghĩa để tìm văn bản dựa trên nghĩa của nó.

- Vectors chủ đề có thể được sử dụng để dự đoán khi nào một bài post trên mạng xã hội là SPAM

- Quan trọng là bạn đã biết cách xoay sở với bài toán hóc búa số chiều (dimensionality) để tìm ra các xấp xỉ trong không gian vector ngữ nghĩa (semantic vector space).

./