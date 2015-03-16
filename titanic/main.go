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
	// train
	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		var linregFuncs []func(passengers []passenger) *linreg.LinearRegression

		linregFuncs = append(linregFuncs, linregSexAge)
		linregFuncs = append(linregFuncs, linregPClassAge)
		linregFuncs = append(linregFuncs, linregPClassSex)

		linregs = trainModel(reader, linregFuncs)
	}
	// test models ...
	if csvfile, err := os.Open(*test); err != nil {
		log.Fatalln(err)
	} else {
		//"data/testModel-SexAge.csv"
		//"data/testModel-PClassAge.csv"
		//"data/testModel-PClassSex.csv"
		reader := csv.NewReader(csvfile)
		testModel(reader, linregs[0], "foo")
	}
}
