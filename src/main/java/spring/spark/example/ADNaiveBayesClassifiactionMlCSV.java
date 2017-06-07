package spring.spark.example;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructField;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import spring.spark.example.model.Advertisement;
/**
 * 나이브 베이지안을 이용한 광고 카테고리 분류. 
 * @author skan
 *
 */
public class ADNaiveBayesClassifiactionMlCSV {
	final static Logger logger = LoggerFactory.getLogger(ADNaiveBayesClassifiactionMlCSV.class);
	
	public static void main(String[] args) throws Exception {
		
		logger.info("ADNaiveBayesClassifiaction Start ~ ~");
		
		System.setProperty("hadoop.home.dir", "D:/example/spark/spark-2.1.0-bin-hadoop2.7/spark-2.1.0-bin-hadoop2.7");
		
		SparkConf sparkConf = new SparkConf().setAppName("ADNaiveBayesClassifiaction").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		SparkSession spark = SparkSession
			      .builder()
			      .appName("ADNaiveBayesClassifiactionMlCSV")
			      .getOrCreate() ;
		
		
		Dataset<Row> df = spark.read()
				.option("header", true)
				.csv("data/ad/adcategory.csv");
				//.format("csv").load("data/ad/adcategory.csv");
		
		// Spring Indexer Model을 이용하여 label 생성  > [카테고리 번호]를 기준으로 label 생성
		//StringIndexer indexer = new StringIndexer()
		//		  .setInputCol("category")
		//		  .setOutputCol("label");
		
		StringIndexerModel indexer = new StringIndexer()
						  .setInputCol("category")
						  .setOutputCol("label").fit(df);
		
		Dataset<Row> indexed = indexer.transform(df);
		indexed.show(20);
		
		StructField inputColSchema = indexed.schema().apply(indexer.getOutputCol());
		System.out.println("StringIndexer will store labels in output column metadata: " +
			    Attribute.fromStructField(inputColSchema).toString() + "\n");
		
		
		
		IndexToString converter = new IndexToString()
				  .setInputCol("label")
				  .setOutputCol("originalCategory");
		Dataset<Row> sentenceData = converter.transform(indexed);
		
		
		
		
		Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
		Dataset<Row> wordsData = tokenizer.transform(sentenceData);

		// 키워드 해싱 
		int numFeatures = 10000;
		HashingTF hashingTF = new HashingTF()
									.setInputCol("words")
									.setOutputCol("rawFeatures")
									.setNumFeatures(numFeatures);
		Dataset<Row> featurizedData = hashingTF.transform(wordsData);
		logger.info("//////////////////////////////////////");
		logger.info("//			hashingTF DATA			  ");
		//featurizedData.show();
		
		// alternatively, CountVectorizer can also be used to get term frequency
		// vectors
		// 빈발 항목에대한 카운터 벡터 생성 
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfModel = idf.fit(featurizedData);

		// 빈발 항목 빈발 벡터 가중치 적용
		logger.info("//////////////////////////////////////");
		logger.info("//			TDIDF - Model DATA			  ");
		Dataset<Row> rescaledData = idfModel.transform(featurizedData);
		rescaledData.show(1000);
		
		// 특정 컬럼만 선택 하여 보기.
		//rescaledData.select("label","words" , "features", "rawFeatures").show();
		
		//logger.info("//			TDIDF - Model DATA			  GROUP");
		//RelationalGroupedDataset aa =  rescaledData.select("label").groupBy("label");
		//Dataset<Row> a =aa.org$apache$spark$sql$RelationalGroupedDataset$$df;
		//a.show(1000);
		
		
		////////////////////////////////////////////////////////////////
		// RANDOM  트레인데이터 60% , 테스트 데이터 40%
		////////////////////////////////////////////////////////////////
		Dataset<Row>[] splits  = rescaledData.randomSplit(new double[] { 0.2, 0.8 });
		Dataset<Row> training = splits[0]; // training set
		Dataset<Row> test = splits[1]; // test set
		
		// create the trainer and set its parameters
		NaiveBayes nb = new NaiveBayes();

		// 학습 모델
		NaiveBayesModel model = nb.fit(training);
		
		// Select example rows to display.
		Dataset<Row> predictions = model.transform(test);
		predictions.show();
		
		// 예측 데이터 결과
		List<Row> listOne = predictions.collectAsList();
		listOne.forEach(prd->{
			logger.info("/////////////////////////////////////");
			double label  = prd.getAs("label");
			//WrappedArray<?> wordAr  = (WrappedArray<?>) prd.get(2);
			//scala.collection.Iterator iter = wordAr.iterator();
			//while (iter.hasNext()) 
			//System.out.println(iter.next());
			//Predef$.MODULE$.wrapString(wordAr);
			//wordAr.toList();
			
			String sentence 		= prd.getAs("text");
			String originalCategory = prd.getAs("originalCategory");
			DenseVector probabilitys  = (DenseVector)prd.getAs("probability");
			
			//logger.info("Test Data 라벨  [{}] 텍스트 =[{}]", label, sentence);
			
			AtomicInteger ai = new AtomicInteger();
			List<Advertisement> advertisements = new ArrayList<>();
			
			Arrays.stream(probabilitys.values())
						//.sorted()
						.forEach( probability-> {
							int seq  = ai.incrementAndGet();
							advertisements.add(new Advertisement((seq-1), label, probability, sentence ,originalCategory));
							//logger.info("우선순위  예측 확율 = {}" ,probability);
						}
					);
			
			
			// List<Advertisement> orderList =advertisements.stream().sorted(
			//		 (a,b) -> (a.getProbability() > a.getProbability())? -1: (a.getProbability() < a.getProbability())? 1:0).collect(Collectors.toList());
			// orderList.forEach( ad -> {
			//	 						logger.info(" 라벨 = {} , 우선순위 예측 확율 = {}" , ad.getOrder(),ad.getProbability());
			//	 });
			 
			// DESC 정렬
			advertisements.stream().sorted(Comparator.comparing(Advertisement :: getProbability ).reversed() ).forEach( ad -> {
				logger.info(" 라벨 = {} ,  우선순위  예측 확율 = {} , words = {}" , ad.getOrder(), ad.getProbability() , ad.getWords()  );
			});
			
		});
				

		// compute accuracy on the test set
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
															.setLabelCol("label")
															.setPredictionCol("prediction")
															.setMetricName("accuracy");
		
		
		double accuracy = evaluator.evaluate(predictions);
		logger.info("AD Data Test set accuracy = {}",  accuracy);
		
		jsc.stop();
	}
}


