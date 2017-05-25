package spring.spark.example;


import java.util.Arrays;
import java.util.List;

// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.Tuple2;
/**
 * ���̺� ���������� �̿��� ���� ī�װ� �з�. 
 * @author skan
 *
 */
public class ADNaiveBayesClassifiaction2 {
	final static Logger logger = LoggerFactory.getLogger(ADNaiveBayesClassifiaction2.class);
	
	public static void main(String[] args) throws Exception {
		
		logger.info("ADNaiveBayesClassifiaction Start ~ ~");
		
		System.setProperty("hadoop.home.dir", "D:/example/spark/spark-2.1.0-bin-hadoop2.7/spark-2.1.0-bin-hadoop2.7");
		
		SparkConf sparkConf = new SparkConf().setAppName("ADNaiveBayesClassifiaction").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		SparkSession spark = SparkSession
			      .builder()
			      .appName("ADNaiveBayesClassifiaction")
			      .getOrCreate() ;
		
		
		// Input data: Each row is a bag of words from a sentence or document.
		List<Row> data = Arrays.asList(
		  RowFactory.create(1.0 , "�ڵ��� ������ ���� ���׿��극���� �����̵�"),
		  RowFactory.create(3.0 , "������̼� ���� GPS ���� ���߱��� �������"),
		  RowFactory.create(3.0 , "������̼� ���� GPS ���� ���߱��� �������"),
		  RowFactory.create(3.0 , "������̼� ���� GPS ���� ���߱��� �������"),
		  RowFactory.create(3.0 , "������̼� ���� GPS ���� ���߱��� �������"),
		  RowFactory.create(3.0 , "������̼� ���� GPS ���� ���߱��� �������"),
		  RowFactory.create(3.0 , "������̼� ���� GPS ���� ���߱��� �������"),
		  RowFactory.create(0.0 , "���� ����� �ƽþƳ� �����װ�"),
		  RowFactory.create(0.0 , "���� ����� �ƽþƳ� �����װ�"),
		  RowFactory.create(1.0 , "�ڵ��� ������ ���� ���׿��극���� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(1.0 , "�ڵ��� ���� ������ �ٵ� �����̵�"),
		  RowFactory.create(0.0 , "Logistic regression models are neat"),
		  RowFactory.create(5.0, "I wish Java could use case classes"),
		  RowFactory.create(4.0, "��Ÿ�� ���̵� ���� �� ��Ƽ �÷��� ��� ���̵�"),
		  RowFactory.create(4.0, "��Ÿ�� ���̵� ���� �� ��Ƽ �÷��� ��� ���̵�"),
		  RowFactory.create(4.0, "��Ÿ�� ���̵� ���� �� ��Ƽ �÷��� ��� ���̵�"),
		  RowFactory.create(4.0, "��Ÿ�� ���̵� ���� �� ��Ƽ �÷��� ��� ���̵�"),
		  RowFactory.create(4.0, "��Ÿ�� ���̵� ���� �� ��Ƽ �÷��� ��� ���̵�")
		  
		);
		StructType schema = new StructType(
				new StructField[] { 
						new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
						new StructField("sentence", DataTypes.StringType, false, Metadata.empty()) });
		Dataset<Row> sentenceData = spark.createDataFrame(data, schema);

		Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
		Dataset<Row> wordsData = tokenizer.transform(sentenceData);

		// Ű���� �ؽ� 
		int numFeatures = 30;
		HashingTF hashingTF = new HashingTF()
									.setInputCol("words")
									.setOutputCol("rawFeatures")
									.setNumFeatures(numFeatures);
		Dataset<Row> featurizedData = hashingTF.transform(wordsData);
		System.out.println("//////////////////////////////////////");
		System.out.println("//			hashingTF DATA			  ");
		//featurizedData.show();
		
		// alternatively, CountVectorizer can also be used to get term frequency
		// vectors
		// ��� �׸񿡴��� ī���� ���� ���� 
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfModel = idf.fit(featurizedData);

		// ��� �׸� ��� ���� ����ġ ����
		System.out.println("//////////////////////////////////////");
		System.out.println("//			TDIDF - Model DATA			  ");
		Dataset<Row> rescaledData = idfModel.transform(featurizedData);
		rescaledData.show();
		//rescaledData.select("label","words" , "features", "rawFeatures").show();
		
		// ���� ����
		//model.save("target/feture/cntVector");
		JavaRDD<LabeledPoint> LabeledPointdata = rescaledData.toJavaRDD().map(row->  {
			double label        	 =  row.getDouble(0);
			// List<Object> str         =  row.getList(2) ;
			Vector vector 			 = (Vector) row.get(3);
			return new LabeledPoint(label, org.apache.spark.mllib.linalg.Vectors.dense(vector.toArray())); 
			}
		);
		
		LabeledPointdata.cache();
		/////////////////////////////////////////////////
		// NaiveBayes �� ����  RUN
		/////////////////////////////////////////////////
		LabeledPoint dataPoint =  LabeledPointdata
													.take(100)
													.get(0);
        System.out.println("dataPoint.features() = " + dataPoint.features());
        System.out.println("dataPoint.label() = " + dataPoint.label());
        
        // 1.
		NaiveBayesModel nbModel = NaiveBayes.train(JavaRDD.toRDD(LabeledPointdata));
		double predictValue = nbModel.predict(dataPoint.features());
		
		Arrays.stream(nbModel.labels()).forEach(label -> {System.out.println("labels = " + label);});
		Arrays.stream(nbModel.pi()).forEach( pi -> {System.out.println("pi= " + pi);});
		System.out.println("[" + dataPoint.label() + "] ���� �з��Ⱚ = " + predictValue);
		
		
		
		
		////////////////////////////////////////////////////////////////
		// RANDOM  Ʈ���ε����� 60% , �׽�Ʈ ������ 40%
		////////////////////////////////////////////////////////////////
		JavaRDD<LabeledPoint>[] tmp = LabeledPointdata.randomSplit(new double[] { 0.6, 0.4 });
		JavaRDD<LabeledPoint> training = tmp[0]; // training set
		JavaRDD<LabeledPoint> test = tmp[1]; // test set
		NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
		JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> 
				{
					// System.out.println(p.features() + "/"  +  p.label());
					return new Tuple2<>(model.predict(p.features()), p.label());
				}
		);
		
		double accuracy = predictionAndLabel.filter(
				pl -> pl._1().equals(pl._2())
		).count() / (double) test.count();
		
		
		// ���� ���� �� Ʃ�÷� Ȯ��
		predictionAndLabel.foreach(datas -> {
			        System.out.println("model="+datas._1() + " label=" + datas._2());
				});
		
		System.out.println("RANDOM ��Ȯ��=" +  accuracy  + "%");
		jsc.stop();
	}
}


