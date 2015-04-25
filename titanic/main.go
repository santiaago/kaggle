package main

import (
	"flag"
	"log"
	"sort"

	"github.com/santiaago/kaggle/data"
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

	linregs, featuresPerModel := trainModels(drTrain)

	mapUsedFeatures := make(map[string][]int)
	for i := 0; i < len(linregs); i++ {
		mapUsedFeatures[linregs[i].Name] = featuresPerModel[i]
	}

	var xTest data.Extractor = NewPassengerTestExtractor()
	var drTest data.Reader = NewPassengerReader(*test, xTest)

	var wTest data.Writer = NewPassengerTestWriter(*test)
	testModels(drTest, wTest, linregs, mapUsedFeatures)

	var rgs = regressions(linregs)
	sort.Sort(rgs)
	log.Printf("best 50 models:\n\n")
	rgs.Print(50)
}
