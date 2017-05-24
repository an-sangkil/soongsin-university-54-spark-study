



package spring.spark.example;


/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// $example on$
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.util.Arrays;

// $example off$
import org.apache.spark.SparkConf;

public class JavaNaiveBayesExample {
	public static void main(String[] args) {
		
		//다음 오류는 Spark 응용 프로그램을 실행하는 동안 classpath에 winutils 바이너리가 누락되어 발생합니다. 
		//Winutils는 Hadoop 생태계의 일부이며 Spark에는 포함되지 않습니다. 예외가 throw 된 후에도 응용 프로그램의 실제 기능이 올바르게 실행될 수 있습니다. 
		//그러나 불필요한 문제를 피하기 위해 제자리에 두는 것이 좋습니다. 오류를 피하려면 winutils.exe 바이너리를 다운로드하여 클래스 패스에 추가하십시오.
		System.setProperty("hadoop.home.dir", "D:/example/spark/spark-2.1.0-bin-hadoop2.7/spark-2.1.0-bin-hadoop2.7");
		
		SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local[*]");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);
		// $example on$
		String path = "data/mllib/sample_libsvm_data.txt";
		JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();
		JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[] { 0.7, 0.3 });
		
		
		JavaRDD<LabeledPoint> training = tmp[0]; // training set
		JavaRDD<LabeledPoint> test = tmp[1]; // test set
		NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
		
		
		Arrays.stream(model.labels()).forEach(label->System.out.println("label = " + label));
		Arrays.stream(model.pi()).forEach(pi->System.out.println("pi = " + pi));
		
		JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> 
																			{
																				System.out.println(p.features() + "/"  +  p.label());
																				return new Tuple2<>(model.predict(p.features()), p.label());
																			}
		
				);
		
		
		
		double accuracy = predictionAndLabel.filter(
					pl -> pl._1().equals(pl._2())
				).count() / (double) test.count();

		System.out.println("정확도=" +  accuracy);
		System.out.println(training.count() );
		
		
		// Save and load model
		
		//model.save(jsc.sc(), "target/tmp/myNaiveBayesModel");
		//NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "target/tmp/myNaiveBayesModel");
		
		// $example off$
		
		//System.out.println("sameModel = " + sameModel);
		jsc.stop();
	}
}
