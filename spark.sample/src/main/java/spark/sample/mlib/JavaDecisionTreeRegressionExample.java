package spark.sample.mlib;

//$example on$
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
//$example off$

class JavaDecisionTreeRegressionExample {

	public static void main(String[] args) {

		// $example on$
		SparkConf sparkConf = new SparkConf().setMaster("local"). setAppName("JavaDecisionTreeRegressionExample");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);

		// Load and parse the data file.
		String datapath = "/Users/skan/temp/dataset/a1a.txt";
		JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
		// Split the data into training and test sets (30% held out for testing)
		JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[] { 0.7, 0.3 });
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];

		// Set parameters.
		// Empty categoricalFeaturesInfo indicates all features are continuous.
		Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
		String impurity = "variance";
		Integer maxDepth = 5;
		Integer maxBins = 32;

		// Train a DecisionTree model.
		final DecisionTreeModel model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, impurity,
				maxDepth, maxBins);

		// Evaluate model on test instances and compute test error
		JavaPairRDD<Double, Double> predictionAndLabel = testData
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
					}
				});
		Double testMSE = predictionAndLabel.map(new Function<Tuple2<Double, Double>, Double>() {
			@Override
			public Double call(Tuple2<Double, Double> pl) {
				Double diff = pl._1() - pl._2();
				return diff * diff;
			}
		}).reduce(new Function2<Double, Double, Double>() {
			@Override
			public Double call(Double a, Double b) {
				return a + b;
			}
		}) / data.count();
		System.out.println("Test Mean Squared Error: " + testMSE);
		System.out.println("Learned regression tree model:\n" + model.toDebugString());

		// Save and load model
		model.save(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
		DecisionTreeModel sameModel = DecisionTreeModel.load(jsc.sc(), "target/tmp/myDecisionTreeRegressionModel");
		// $example off$
		
		
		
	}
}