package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/santiaago/ml/data"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")

	trainLinreg = flag.Bool("linreg", false, "train linear regressions.")

	trainLinregSpecific     = flag.Bool("linregspec", false, "train specific linear regressions.")
	trainLinregCombinations = flag.Bool("linregcomb", false, "train linear regressions combinations.")
	trainLinregTransforms   = flag.Bool("linregtrans", false, "train all linear regressions with transformations.")
	trainLinregRegularized  = flag.Bool("linregReg", false, "train all linear regressions with regularization.")

	trainLogreg = flag.Bool("logreg", false, "train logistic regressions.")

	trainLogregSpecific     = flag.Bool("logregspec", false, "train specific logistic regressions.")
	trainLogregCombinations = flag.Bool("logregcomb", false, "train logistic regressions combinations.")
	trainLogregTransforms   = flag.Bool("logregtrans", false, "train all logistic regressions with transformations.")
	trainLogregRegularized  = flag.Bool("logregReg", false, "train all logistic regressions with regularization.")

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

	writeEinRanking(models, "ranking.ein.md")
	// todo(santiaago): too slow
	writeEcvRanking(models, "ranking.ecv.md")

	if *einRank {
		models.TopEin(25)
	}

	if *ecvRank {
		models.TopEcv(25)
	}
}
