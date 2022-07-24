using Microsoft.ML;
using Microsoft.ML.Data;
using ML_HelloWorld.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML_HelloWorld.Ml
{
    public static class MlSteps
    {

        private static string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", @"..\..\..\..\..\Data\sentiment labelled sentences\yelp_labelled.txt");

        public static TrainTestData LoadData(MLContext mlContext)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            //load data as SentimentData; theht file do not contains header.
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

            //split data 
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            return splitDataView;
        }
        public static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
        {
            Console.ForegroundColor = ConsoleColor.Green;

            //convert from text data into number values
            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Create and Train the Model ===============");
            //training
            var model = estimator.Fit(splitTrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            return model;

        }

        public static void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            Console.ForegroundColor = ConsoleColor.Blue;

            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            //using test data for testing model
            IDataView predictions = model.Transform(splitTestSet);
            // check score with prediction values and original values
            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

        }

        public static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
        {

            Console.ForegroundColor = ConsoleColor.Cyan;

            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "This was a very bad steak"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();
        }

        public static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
        {


            //create simple additional data for testing
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "This was a horrible meal"
                },
                new SentimentData
                {
                    SentimentText = "I love this spaghetti."
                }
            };
            Console.ForegroundColor = ConsoleColor.Magenta;
            //convert data from enumerable as numerical data
            IDataView batchComments = mlContext.Data.LoadFromEnumerable<SentimentData>(sentiments);

            //again testing
            IDataView predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();

            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }

        public static void Testing(MLContext mlContext, ITransformer model, string comment)
        {
            IEnumerable<SentimentData> testdataList =new List<SentimentData>(){ new SentimentData { SentimentText =comment}  };
            IDataView TestData = mlContext.Data.LoadFromEnumerable(testdataList);

            var TestingResult = model.Transform(TestData);
            var result = mlContext.Data.CreateEnumerable<SentimentPrediction>(TestingResult, reuseRowObject: false);

            foreach (var Variable in result)
            {
                Console.WriteLine($"Sentiment: {Variable.SentimentText} | Prediction: {(Convert.ToBoolean(Variable.Prediction) ? "Positive" : "Negative")} | Probability: {Variable.Probability} ");
            }
              
        }
    }
}
