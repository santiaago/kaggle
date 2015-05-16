package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")

	importPath      = flag.String("ipath", "models.json", "path to a json array with models to use description.")
	canImportModels = flag.Bool("i", false, "defines if the program should import the models defined in ipath")

	exportPath      = flag.String("epath", "usedModels.json", "json array with the description of the trained models.")
	canExportModels = flag.Bool("e", false, "defines if the program should export the used models defined in epath")

	tempPath = flag.String("temp", "data/temp/", "path of temp folder where all model results and rankings will be written.")

	trainLinreg = flag.Bool("linreg", false, "train linear regressions.")
	trainLogreg = flag.Bool("logreg", false, "train logistic regressions.")
	trainSvm    = flag.Bool("svm", false, "train support vector machines.")

	trainSpecific      = flag.Bool("specific", false, "train specific models.")
	combinations       = flag.Int("comb", 0, "number of features to try with all combinations.")
	trainTransforms    = flag.Bool("trans", false, "train models with transformations.")
	transformDimension = flag.Int("dim", 0, "dimension of transformation.")
	trainRegularized   = flag.Bool("reg", false, "train models with regularization.")

	svmKRange = flag.Int("svmKRange", 1, "range of number of block size that should be try for the svm pegasos algorithm. If k = 10, we will try all values from 1 to k")
	svmK      = flag.Int("svmK", 1, "number of block size that should be try for the svm pegasos algorithm.")
	svmLambda = flag.Float64("svmL", 0.001, "lambda, regularization parameter.")

	rankEin = flag.Bool("rankEin", false, "writes a ranking.ein.md file with the in sample ranking of all processed models.")
	rankEcv = flag.Bool("rankEcv", false, "writes a ranking.ecv.md file with the cross validation ranking of all processed models.")

	verbose = flag.Bool("v", false, "verbose: print additional output")
)

func init() {
	log.SetFlags(log.Ltime | log.Ldate | log.Lshortfile)
}

func main() {
	flag.Parse()

	var xTrain data.Extractor
	var drTrain data.Reader
	xTrain = NewPassengerTrainExtractor()
	drTrain = NewPassengerReader(*train, xTrain)

	models := trainModels(drTrain)

	if len(models) == 0 {
		if *verbose {
			fmt.Println("no models found.")
		}
		return
	}

	var xTest data.Extractor
	var wTest data.Writer
	var drTest data.Reader

	xTest = NewPassengerTestExtractor()
	wTest = NewPassengerTestWriter(*test)
	drTest = NewPassengerReader(*test, xTest)

	testModels(drTest, wTest, models)

	rank(models)

	if *canExportModels {
		exportModels(models, *exportPath)
	}

}

func rank(models ml.ModelContainers) {
	if *verbose {
		fmt.Println("Start ranking models")
	}
	if *rankEin {
		if *verbose {
			fmt.Println("Start ranking models by Ein")
		}
		writeEinRanking(models, "ranking.ein.md")
		models.TopEin(25)
		if *verbose {
			fmt.Println("Done ranking models by Ein")
		}
	}

	if *rankEcv {
		if *verbose {
			fmt.Println("Start ranking models by Ecv")
		}
		writeEcvRanking(models, "ranking.ecv.md")
		models.TopEcv(25)
		if *verbose {
			fmt.Println("Done ranking models by Ecv")
		}
	}
	if *verbose {
		fmt.Println("Done ranking models")
	}
}
