package spring.spark.example;

//$example on$
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
//$example off$

public class TextConvtorToVector2 {
	public static void main(String[] args) {
		
		System.setProperty("hadoop.home.dir", "D:/example/spark/spark-2.1.0-bin-hadoop2.7/spark-2.1.0-bin-hadoop2.7");
		SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		
		SparkSession spark = SparkSession.builder().appName("JavaCountVectorizerExample").getOrCreate();

		// $example on$
		// Input data: Each row is a bag of words from a sentence or document.
		List<Row> data = Arrays.asList(
				RowFactory.create(0.0,Arrays.asList("a", "b", "c")),
				RowFactory.create(1.0,Arrays.asList("a", "b", "b", "c", "a")),
				RowFactory.create(1.0,Arrays.asList("a", "b", "b", "c", "a")),
				RowFactory.create(1.0,Arrays.asList("a", "b", "b", "c", "a")),
				RowFactory.create(1.0,Arrays.asList("a", "b", "b", "c", "a")),
				RowFactory.create(1.0,Arrays.asList("a", "b", "b", "c", "a")),
				RowFactory.create(1.0,Arrays.asList("a", "b", "b", "c", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a")),
				RowFactory.create(2.0,Arrays.asList("d", "d", "a", "a", "a"))
				);
		
		
		StructType schema = new StructType(new StructField[] {
				new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
				new StructField("word", new ArrayType(DataTypes.StringType, true), false, Metadata.empty()) });
		Dataset<Row> df = spark.createDataFrame(data, schema);

		// fit a CountVectorizerModel from the corpus
		CountVectorizerModel cvModel = new CountVectorizer()
												.setInputCol("word")
												.setOutputCol("feature")
												.setVocabSize(5)
												.setMinDF(1).fit(df);

		// alternatively, define CountVectorizerModel with a-priori vocabulary
		CountVectorizerModel cvm = new CountVectorizerModel(new String[] { "a", "b", "c" , "d" }).setInputCol("word")
				.setOutputCol("feature");
		 Dataset<Row> features =  cvModel.transform(df);
		 features.show();
		 
		 
		 System.out.println("firstData= " + cvModel.transform(df).select("feature").first()  );
		// $example off$

		spark.stop();
	}
}
