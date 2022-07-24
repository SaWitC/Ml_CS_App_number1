using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using ML_HelloWorld.Ml;
using ML_HelloWorld.Models;
using static Microsoft.ML.DataOperationsCatalog;

namespace ML_HelloWorld
{
    class Program
    {
        static void Main(string[] args)
        {  
            var mlContext = new MLContext();

            Console.ForegroundColor = ConsoleColor.Yellow; Console.WriteLine("step 1 load data");
            TrainTestData splitDataView = MlSteps.LoadData(mlContext);
            //var piplane = MlContext.Transforms.Conversion.MapValueToKey(null);
            Console.ForegroundColor = ConsoleColor.Yellow; Console.WriteLine("step 2"); 
            ITransformer model = MlSteps.BuildAndTrainModel(mlContext, splitDataView.TrainSet);
            Console.ForegroundColor = ConsoleColor.Yellow; Console.WriteLine("step 3");

            MlSteps.Evaluate(mlContext, model, splitDataView.TestSet);
            Console.ForegroundColor = ConsoleColor.Yellow; Console.WriteLine("step 4");

            MlSteps.UseModelWithSingleItem(mlContext, model);
            Console.ForegroundColor = ConsoleColor.Yellow; Console.WriteLine("step 5");

            MlSteps.UseModelWithBatchItems(mlContext, model);

            Console.ForegroundColor = ConsoleColor.Yellow; Console.WriteLine("step 6");

            Console.WriteLine("Write your comment");
            string comment = Console.ReadLine();
            MlSteps.Testing(mlContext,model,comment);

            Console.ReadKey();
        }

        
    }
}
