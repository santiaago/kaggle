package main

import (
	"encoding/csv"
	"flag"
	"log"
	"os"
	"sort"
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

	dr := NewReader(*train)

	linregs, featuresPerModel := trainModels(dr)

	mapUsedFeatures := make(map[string][]int)
	for i := 0; i < len(linregs); i++ {
		mapUsedFeatures[linregs[i].Name] = featuresPerModel[i]
	}

	testModels(*test, linregs, mapUsedFeatures)

	var rgs = regressions(linregs)
	sort.Sort(rgs)
	log.Printf("\n\n\n")
	rgs.Print(50)
}

func NewReader(file string) Reader {
	var r *csv.Reader
	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		r = csv.NewReader(csvfile)
	}

	return Reader{
		NewPassengerExtractor(r),
		NewPassengerCleaner(),
	}
}
