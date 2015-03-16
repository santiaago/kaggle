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

	var linreg *linreg.LinearRegression
	// train
	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		linreg = trainModel(reader)
	}
	// test ...
	if csvfile, err := os.Open(*test); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		testModel(reader, linreg)
	}
}
