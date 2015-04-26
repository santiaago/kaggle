package main

import (
	"flag"
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

	var xTrain data.Extractor = NewPassengerTrainExtractor()
	var drTrain data.Reader = NewPassengerReader(*train, xTrain)

	models := trainModels(drTrain)

	var xTest data.Extractor = NewPassengerTestExtractor()
	var drTest data.Reader = NewPassengerReader(*test, xTest)

	var wTest data.Writer = NewPassengerTestWriter(*test)
	testModels(drTest, wTest, models)

	models.PrintTop(50)
}
