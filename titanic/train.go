package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

func trainModels(r *csv.Reader, linregFuncs []func(passengers []passenger) *linreg.LinearRegression) []*linreg.LinearRegression {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}
	var linregs []*linreg.LinearRegression
	for _, f := range linregFuncs {
		linregs = append(linregs, f(passengers))
	}
	return linregs
}

func trainMetaModels(file string) (linregs []*linreg.LinearRegression, linregsNames []string) {

	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		metaLinregFuncs := []func(passengers []passenger) []*linreg.LinearRegression{
			linregVectorsOf2,
			linregVectorsOf3,
			linregVectorsOf4,
			linregVectorsOf5,
			linregVectorsOf6,
			linregVectorsOf7,
		}
		linregsOf := trainModelsMeta(csv.NewReader(csvfile), metaLinregFuncs)
		for i := 0; i < len(linregsOf); i++ {
			name := fmt.Sprintf("data/temp/testModel-V-%d", i)
			linregsNames = append(linregsNames, name)
		}
		linregs = append(linregs, linregsOf...)
	}
	return
}

func trainModelsMeta(r *csv.Reader, metaLinregFuncs []func(passengers []passenger) []*linreg.LinearRegression) []*linreg.LinearRegression {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}
	var linregs []*linreg.LinearRegression
	for _, f := range metaLinregFuncs {
		linregs = append(linregs, f(passengers)...)
	}
	return linregs
}
