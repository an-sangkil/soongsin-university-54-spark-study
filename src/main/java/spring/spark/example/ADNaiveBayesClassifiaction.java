package spring.spark.example;


import java.util.Arrays;
import java.util.List;

// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class ADNaiveBayesClassifiaction {
	public static void main(String[] args) throws Exception {
		
		System.setProperty("hadoop.home.dir", "D:/example/spark/spark-2.1.0-bin-hadoop2.7/spark-2.1.0-bin-hadoop2.7");
		
		SparkConf sparkConf = new SparkConf().setAppName("ADNaiveBayesClassifiaction").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		SparkSession spark = SparkSession
			      .builder()
			      .appName("ADNaiveBayesClassifiaction")
			      .getOrCreate() ;
		
		
		// Input data: Each row is a bag of words from a sentence or document.
		List<Row> data = Arrays.asList(
		  RowFactory.create(1.0 , Arrays.asList("자동차 오락실 게임 리그오브레전드 아케이드".split(" "))),
		  RowFactory.create(2.0 , Arrays.asList("I wish Java could use case classes".split(" "))),
		  RowFactory.create(3.0 , Arrays.asList("Logistic regression models are neat".split(" ")))
		);
		StructType schema = new StructType(new StructField[]{
				new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
				new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
		});
		Dataset<Row> documentDF = spark.createDataFrame(data, schema);

		// Learn a mapping from words to Vectors.
		Word2Vec word2Vec = new Word2Vec()
		  .setInputCol("text")
		  .setOutputCol("result")
		  .setVectorSize(5)
		  .setMinCount(0);

		Word2VecModel model = word2Vec.fit(documentDF);
		Dataset<Row> result = model.transform(documentDF);
		result.show();
		
		// 벡더 터장
		//model.save("target/feture/cntVector");
		System.out.println("getDouble = " + model.getVectors());
		
		JavaRDD<LabeledPoint> LabeledPointdata = result.toJavaRDD().map(row->  {
			double label         = (double) row.get(0);
			List<Object> str         =  row.getList(1) ;
			Vector vector = (Vector) row.get(2);
			System.out.println(vector.toArray());
			org.apache.spark.ml.linalg.DenseVector features = (DenseVector) row.get(2);
			
			return new LabeledPoint(label, org.apache.spark.mllib.linalg.Vectors.dense(vector.toArray())); 
			}
		);
		
		NaiveBayesModel nbModel = NaiveBayes.train(JavaRDD.toRDD(LabeledPointdata));
		
		
		/*
		JavaRDD<String> rawData = jsc.textFile("data/ad/sample.tsv");
		JavaRDD<String[]> records = rawData.map(str->   {
			return str.split(" ");
		});
		JavaRDD<LabeledPoint> LabeledPointdata = records.map(arr->  {
				int label = Integer.parseInt(arr[0]);
				double[] features = new double[1];
				
				for (String string : arr) {
					//System.out.print(array = arr[i]);
				}
				
				//return new LabeledPoint(label, Ve); 
				return new LabeledPoint(label, (Vector) Vectors.dense(features)); 
			}
		);
		
		//System.out.println("data="+LabeledPointdata.count());
		NaiveBayesModel nbModel = NaiveBayes.train(JavaRDD.toRDD(LabeledPointdata));
		*/

		jsc.stop();
	}
}


