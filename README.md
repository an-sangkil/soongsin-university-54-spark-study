# SungSil-54-Spark-Study
숭실대 정보과학대학원  54기 논문 공부방

### 단어추출
1. pre-processing

#### 단어 > 벡터데이터변환
##### 2. Featurize  
	데이터를 기계 학습 알고리즘이 이해 할 수 있는 숫자로 변환.
##### 2.1. Term Frequency 
	TF => 특정 단어가 (문서에) 몇번 등장 했는지 나나내는값 
	Spark CountVectorizer 이용하여 TF Vector 를 구한다.
	---------------------------------------
	|뉴스 ID | 스마트폰[0] | 공장[1] | 날찌 [2] | TF Vector 
	  뉴스1	       4           1         0        [4,1,0]
	  뉴스2	       1           0         3        [1,0,3]
	  뉴스3	       2           0         1        [2,0,1]
	  뉴스4	       3           1         0        [3,1,0]
	  ...

##### 2.3 TF-IDF (Inverse Documet Frequency) 
	TF  : 어떤 단어가 한 문서에 자주 나온다면, 그 단어는 해당 문서를 대표한다
	IDF : 하지만, 다른 문서에도 자주 나오는 단어라면 아니다” • IDF를 통해 문서 전반적으로 많이 나오는 단어의 TF 값 을 낮춰줍니다
	TF-IDF 는 문서의 중요단어를 나타내는 통계적 수치 

#### 벡터>학습>모델
##### 3. Training
	텍스트 데이터에 성능이 좋은 나이브 베이지안 이용 
	3.1 NaiveBayesTF
	3.2 NaiveBayseTFIDF
### 모델의 평가
##### 4. Evaluation




## JAVA SAMPLE
* http://spark.apache.org/docs/latest/quick-start.html
* http://spark.apache.org/examples.html

## 읽어보자 

* http://hyunje.com/data%20analysis/2016/02/01/twitter-analysis/
* http://sonsworld.tistory.com/6
* http://www.slideshare.net/RetrieverJo/pmi-twitter-57723391?ref=http://hyunje.com/data%20analysis/2016/02/01/twitter-analysis/
* http://engineering.vcnc.co.kr/2015/05/data-analysis-with-spark/
* https://speakerdeck.com/vcnc/spark-plus-s3-plus-r3-eul-iyonghan-deiteo-bunseog-siseutem-mandeulgi
* http://readme.skplanet.com/?p=12465
