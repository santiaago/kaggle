package main

import (
	"encoding/csv"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

// trainModels returns:
// * an array of trained LinearRegression models.
// * an array of names each one corresponds to a linearRegression model.
// It uses the file passed as param as training data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// We return an array of all the linear regression models trained.
func trainModels(file string) (linregs []*linreg.LinearRegression) {

	linregs = trainSpecificModels(file)
	linregsByComb := trainModelsByFeatureCombination(file)

	linregs = append(linregs, linregsByComb...)
	return
}

// trainSpecificModels trains the following models:
// * linregSexAge
// * linregPClassAge
// * linregPClassSex
// * linregSexAgePClass
// It returns an array of all the linear regression models trained.
func trainSpecificModels(file string) (linregs []*linreg.LinearRegression) {

	funcs := []func(passengers []passenger) *linreg.LinearRegression{
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		linregs = trainModelsByFuncs(csv.NewReader(csvfile), funcs)
	}
	return
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// * an array pf the names of each linearRegression model.
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(file string) (linregs []*linreg.LinearRegression) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		funcs := linregVectorsOfInterval()
		linregsOf := trainModelsByMetaFuncs(csv.NewReader(csvfile), funcs)

		linregs = append(linregs, linregsOf...)
	}
	return
}

// trainModelsByFuncs returns an array of linear regression models with respect
// to an array of functions passed as arguments.
// Those function takes as argument the passengers data and return a linear
// regression model.
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

// trainModelsByMetaFuncs returns an array of linear regression models with
// respect to an array of linear regression functions passed as arguments.
// Those functions takes as argument the passengers data and
// return an array of linear regression model.
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
		linregs = append(linregs, f(passengers)...)
	}

	return linregs
}
