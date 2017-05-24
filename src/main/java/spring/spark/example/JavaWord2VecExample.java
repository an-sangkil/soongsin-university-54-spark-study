package spring.spark.example;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;


public class JavaWord2VecExample {
	
	public static void main(String args[]) {
		
		System.setProperty("hadoop.home.dir", "D:/example/spark/spark-2.1.0-bin-hadoop2.7/spark-2.1.0-bin-hadoop2.7");
		
		SparkConf sparkConf = new SparkConf().setAppName("JavaWord2VecExample").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		SparkSession spark = SparkSession
			      .builder()
			      .appName("JavaWord2VecExample")
			      .getOrCreate() ;


		// Input data: Each row is a bag of words from a sentence or document.
		List<Row> data = Arrays.asList(
		  RowFactory.create(1.0 , Arrays.asList("Hi I heard about Spark".split(" "))),
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

		for (Row row : result.collectAsList()) {
			List<String> text = row.getList(1);
			Vector vector = (Vector) row.get(2);
			System.out.println("Text: " + text + " => \nVector: " + vector + "\n");
		}
		
		
	}

}
