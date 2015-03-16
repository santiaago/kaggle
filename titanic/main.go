package main

import (
	"encoding/csv"
	"flag"
	"fmt"
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

	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		metaLinregFuncs := []func(passengers []passenger) []*linreg.LinearRegression{
			//linregVectorsOf2,
			//linregVectorsOf3,
			//linregVectorsOf4,
			//linregVectorsOf5,
			//linregVectorsOf6,
			linregVectorsOf7,
		}
		linregsOf := trainModelsMeta(csv.NewReader(csvfile), metaLinregFuncs)
		for i := 0; i < len(linregsOf); i++ {
			name := fmt.Sprintf("data/temp/testModel-V-%d", i)
			linregsNames = append(linregsNames, name)
		}
		linregs = append(linregs, linregsOf...)
	}

	// test models ...
	//testModels(*test, linregs, linregsNames)
}

func testModels(file string, linregs []*linreg.LinearRegression, linregsNames []string) {
	for i := 0; i < len(linregs); i++ {
		if csvfile, err := os.Open(file); err != nil {
			log.Fatalln(err)
		} else {
			reader := csv.NewReader(csvfile)
			fmt.Println(linregs[i].Wn)
			testModel(reader, linregs[i], linregsNames[i])

		}
	}
}
