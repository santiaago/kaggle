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

	// writeEinRanking(models, "ranking.ein.md")
	// todo(santiaago): too slow
	// writeEcvRanking(models, "ranking.ecv.md")

	models.PrintTop(500)
}
