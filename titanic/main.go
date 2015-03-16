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
		reader := csv.NewReader(csvfile)

		linregFuncs := []func(passengers []passenger) *linreg.LinearRegression{
			linregSexAge,
			linregPClassAge,
			linregPClassSex,
		}

		linregsNames = []string{
			"data/testModel-SexAge.csv",
			"data/testModel-PClassAge.csv",
			"data/testModel-PClassSex.csv",
		}

		linregs = trainModels(reader, linregFuncs)
	}

	// test models ...
	testModels(*test, linregs, linregsNames)
}

func testModels(file string, linregs []*linreg.LinearRegression, linregsNames []string) {
	for i := 0; i < len(linregs); i++ {
		if csvfile, err := os.Open(file); err != nil {
			log.Fatalln(err)
		} else {
			reader := csv.NewReader(csvfile)
			testModel(reader, linregs[i], linregsNames[i])

		}
	}
}
