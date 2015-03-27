package main

import (
	"encoding/csv"
	"flag"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()

	var linregs []*linreg.LinearRegression
	var linregsNames []string

	// train model
	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		linregFuncs := []func(passengers []passenger) *linreg.LinearRegression{
			linregSexAge,
			linregPClassAge,
			linregPClassSex,
			linregSexAgePClass,
		}
		linregsNames = []string{
			"data/temp/testModel-SexAge.csv",
			"data/temp/testModel-PClassAge.csv",
			"data/temp/testModel-PClassSex.csv",
			"data/temp/testModel-SexAgePClass.csv",
		}
		linregs = trainModels(csv.NewReader(csvfile), linregFuncs)
	}
	linregs2, linregsNames2 := trainMetaModels(*train)
	linregs = append(linregs, linregs2...)
	linregsNames = append(linregsNames, linregsNames2...)

	testModels(*test, linregs, linregsNames)
}
