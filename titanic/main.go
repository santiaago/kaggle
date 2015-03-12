package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
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
		for i := 1; i < len(rawData); i++ {
			p := passengerFromLine(rawData[i])
			fmt.Printf("%+v\n", p)
		}
	}
}

func passengerFromLine(line []string) passenger {
	survived, err := strconv.ParseBool(line[1])
	if err != nil {
		survived = false
	}
	age, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		age = 33
	}

	var embarked string
	if len(line) > 11 {
		embarked = line[11]
	}

	p := passenger{
		line[0],
		survived,
		line[2],
		line[3],
		line[4],
		int(age),
		line[6],
		line[7],
		line[8],
		line[9],
		line[10],
		embarked,
	}
	return p
}
