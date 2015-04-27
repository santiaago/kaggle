package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
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
	// todo(santiaago): refactor this.
	// regModels := regularizedModels(models)

	for _, m := range models {
		if m == nil {
			continue
		}
		lookupModelWithRegularization(m.Model.(*linreg.LinearRegression))
	}

	var xTest data.Extractor = NewPassengerTestExtractor()
	var drTest data.Reader = NewPassengerReader(*test, xTest)

	var wTest data.Writer = NewPassengerTestWriter(*test)

	if len(models) == 0 {
		fmt.Println("no models found.")
		return
	}

	testModels(drTest, wTest, models)
	models.PrintTop(50)
}
