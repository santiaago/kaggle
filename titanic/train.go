package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

// trainModels using the file passed as param as training data.
// trains multiple models using different techniques.
// We return an array of all the linear regression models trained
// and the respective names.
func trainModels(file string) (linregs []*linreg.LinearRegression, names []string) {

	linregs, names = trainSpecificModels(file)

	linregsByComb, namesByComb := trainModelsByFeatureCombination(*train)
	linregs = append(linregs, linregsByComb...)
	names = append(names, namesByComb...)

	return linregs, names
}

// trainSpecificModels trains the following models:
// * linregSexAge
// * linregPClassAge
// * linregPClassSex
// * linregSexAgePClass
// We return an array of all the linear regression models trained
// and the respective names.
func trainSpecificModels(file string) (linregs []*linreg.LinearRegression, names []string) {

	funcs := []func(passengers []passenger) *linreg.LinearRegression{
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}

	names = []string{
		"data/temp/testModel-SexAge.csv",
		"data/temp/testModel-PClassAge.csv",
		"data/temp/testModel-PClassSex.csv",
		"data/temp/testModel-SexAgePClass.csv",
	}

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		linregs = trainModelsByFuncs(csv.NewReader(csvfile), funcs)
	}

	return linregs, names
}

// trainModelsByFeatureCombination makes a model for all combinations of
// features present in the data.
// We return an array of all the linear regression models trained
// and the respective names.
func trainModelsByFeatureCombination(file string) (linregs []*linreg.LinearRegression, names []string) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		funcs := linregVectorsOfInterval()
		linregsOf := trainModelsByMetaFuncs(csv.NewReader(csvfile), funcs)
		for i := 0; i < len(linregsOf); i++ {
			name := fmt.Sprintf("data/temp/testModel-V-%d", i)
			names = append(names, name)
		}

		linregs = append(linregs, linregsOf...)
	}
	return
}

// trainModelsByFuncs creates an array of linear regression models with respect
// to an array of linear regression functions passed as arguments.
// Those function takes as argument the passengers data and
// returns a linear regression model.
func trainModelsByFuncs(r *csv.Reader, funcs []func(passengers []passenger) *linreg.LinearRegression) []*linreg.LinearRegression {
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
	for _, f := range funcs {
		linregs = append(linregs, f(passengers))
	}
	return linregs
}

// trainModelsByMetaFuncs creates an array of linear regression models with respect
// to an array of linear regression functions passed as arguments.
// Those functions takes as argument the passengers data and
// returns an array of linear regression model.
func trainModelsByMetaFuncs(r *csv.Reader, metaLinregFuncs []func(passengers []passenger) []*linreg.LinearRegression) []*linreg.LinearRegression {
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
		fmt.Println(len(f(passengers)))
		linregs = append(linregs, f(passengers)...)
	}
	fmt.Println(len(linregs))
	return linregs
}
