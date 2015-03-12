package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()

	// train
	if csvfile, err := os.Open(*test); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		trainModel(reader)
	}
	// test ...
}

func trainModel(r *csv.Reader) {
	if rawData, err := r.ReadAll(); err != nil {
		log.Fatalln(err)
	} else {
		//passengers := []passenger{}
		for i := 1; i < 3; /*len(rawData)*/ i++ {
			fmt.Println(rawData[i][0], rawData[i][3])
			//p := passenger{rawData[i]}
		}
	}
}
