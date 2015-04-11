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

	var pTrainX PassengerTrainExtractor = NewPassengerTrainExtractor()
	var dr data.Reader = NewPassengerReader(*train, pTrainX)

	linregs, featuresPerModel := trainModels(dr)

	mapUsedFeatures := make(map[string][]int)
	for i := 0; i < len(linregs); i++ {
		mapUsedFeatures[linregs[i].Name] = featuresPerModel[i]
	}

	// var testDr data.Reader = NewPassengerReader(*test)
	testModels(*test, linregs, mapUsedFeatures)

	var rgs = regressions(linregs)
	sort.Sort(rgs)
	log.Printf("\n\n\n")
	rgs.Print(50)
}
