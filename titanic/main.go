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

	trainLinreg = flag.Bool("linreg", false, "train linear regressions.")

	trainLinregSpecific    = flag.Bool("linregspec", false, "train specific linear regressions.")
	trainLinregTransforms  = flag.Bool("linregtrans", false, "train linear regressions with transformations.")
	trainLinregRegularized = flag.Bool("linregReg", false, "train all linear regressions with regularization.")

	trainLogreg = flag.Bool("logreg", false, "train logistic regressions.")

	trainLogregSpecific    = flag.Bool("logregspec", false, "train specific logistic regressions.")
	trainLogregTransforms  = flag.Bool("logregtrans", false, "train all logistic regressions with transformations.")
	trainLogregRegularized = flag.Bool("logregReg", false, "train all logistic regressions with regularization.")

	transformDimension = flag.Int("dim", 0, "dimension of transformation.")
	combinations       = flag.Int("comb", 0, "number of features to try with all combinations.")

	einRank = flag.Bool("einRank", false, "write a ranking.ein.md file with the in sample ranking of all processed models.")
	ecvRank = flag.Bool("ecvRank", false, "write a ranking.ecv.md file with the cross validation ranking of all processed models.")
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
		fmt.Println("no models found.")
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
}

func rank(models ml.ModelContainers) {
	if *einRank {
		writeEinRanking(models, "ranking.ein.md")
		models.TopEin(25)
	}

	if *ecvRank {
		writeEcvRanking(models, "ranking.ecv.md")
		models.TopEcv(25)
	}
}
