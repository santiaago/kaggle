package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

// trainModels returns:
// * an array of trained LinearRegression models.
// It uses the file passed as param as training data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// We return an array of all the linear regression models trained.
func trainModels(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	linregs, usedFeaturesSpecific := trainSpecificModels(file)
	linregsByComb, usedFeaturesByComb := trainModelsByFeatureCombination(file)
	linregsWithTransform, usedFeaturesWithTransform := trainModelsWithTransform(file)

	usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesSpecific...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesByComb...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesWithTransform...)

	linregs = append(linregs, linregsByComb...)
	linregs = append(linregs, linregsWithTransform...)
	return
}

// trainSpecificModels trains the following models:
// * linregSexAge
// * linregPClassAge
// * linregPClassSex
// * linregSexAgePClass
// It returns an array of all the linear regression models trained.
func trainSpecificModels(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	funcs := []func(passengers []passenger) (*linreg.LinearRegression, []int){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		linregs, usedFeaturesPerModel = trainModelsByFuncs(csv.NewReader(csvfile), funcs)
	}
	return
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		funcs := linregVectorsOfInterval()
		linregsOf, usedFeaturesOf := trainModelsByMetaFuncs(csv.NewReader(csvfile), funcs)

		linregs = append(linregs, linregsOf...)
		usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesOf...)
	}
	return
}

// trainModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainModelsWithTransform(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	toTry := []int{
		passengerIndexPclass,
		passengerIndexSex,
		passengerIndexAge,
		passengerIndexSibSp,
		passengerIndexParch,
		passengerIndexTicket,
		passengerIndexFare,
		passengerIndexCabin,
		passengerIndexEmbarked,
	}

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		funcs2D := []func([]float64) []float64{
			transform2D1,
			transform2D2,
			transform2D3,
			transform2D4,
			transform2D5,
			transform2D6,
			transform2D7,
		}

		d := 2
		lrWithTransform, ufWithTransform := trainModelsWithNDTransformFuncs(csv.NewReader(csvfile), funcs2D, toTry, d)

		linregs = append(linregs, lrWithTransform...)
		usedFeaturesPerModel = append(usedFeaturesPerModel, ufWithTransform...)
	}
	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		funcs3D := []func([]float64) []float64{
			transform3D1,
			transform3D2,
			transform3D3,
			transform3D4,
			transform3D5,
			transform3D6,
			transform3D7,
		}
		d := 3
		lrWithTransform3D, ufWithTransform3D := trainModelsWithNDTransformFuncs(csv.NewReader(csvfile), funcs3D, toTry, d)

		linregs = append(linregs, lrWithTransform3D...)
		usedFeaturesPerModel = append(usedFeaturesPerModel, ufWithTransform3D...)

	}
	return
}

// trainModelsWithNDTransformFuncs returns
//   * an array of linearRegression models
//   * an array of used features vectors
// Models are created as follows:
// Data is filtered with respect to the 'candidateFeatures' vector and the 'dimention' param.
// We generate a vector of combinations of the candidateFeatures vector.
// Each combination has the size of the size of 'dimention'.
// Each (combination, transform function) pair is a specific model.
func trainModelsWithNDTransformFuncs(r *csv.Reader, funcs []func([]float64) []float64, candidateFeatures []int, dimention int) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil, nil
	}

	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}
	data := prepareData(passengers)

	combs := combinations(candidateFeatures, dimention)
	for _, comb := range combs {

		filteredData := filter(data, comb)

		index := 0
		for _, f := range funcs {
			linreg := linreg.NewLinearRegression()
			linreg.Name = fmt.Sprintf("%dD %v transformed %d", dimention, comb, index)
			linreg.InitializeFromData(filteredData)
			linreg.TransformFunction = f
			linreg.ApplyTransformation()
			if err := linreg.Learn(); err == nil {
				fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
				linregs = append(linregs, linreg)
				usedFeaturesPerModel = append(usedFeaturesPerModel, comb)
				index++
			}
		}
	}
	return
}

// trainModelsByFuncs returns an array of linear regression models with respect
// to an array of functions passed as arguments.
// Those function takes as argument the passengers data and return a linear
// regression model.
func trainModelsByFuncs(r *csv.Reader, funcs []func(passengers []passenger) (*linreg.LinearRegression, []int)) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil, nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}

	for _, f := range funcs {
		linreg, usedFeatures := f(passengers)
		linregs = append(linregs, linreg)
		usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeatures)
	}
	return
}

// trainModelsByMetaFuncs returns an array of linear regression models with
// respect to an array of linear regression functions passed as arguments.
// Those functions takes as argument the passengers data and
// return an array of linear regression model.
func trainModelsByMetaFuncs(r *csv.Reader, metaLinregFuncs []func(passengers []passenger) ([]*linreg.LinearRegression, [][]int)) ([]*linreg.LinearRegression, [][]int) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil, nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}
	var linregs []*linreg.LinearRegression
	var usedFeatures [][]int
	for _, f := range metaLinregFuncs {
		linreg, usedFeaturesPerModel := f(passengers)
		usedFeatures = append(usedFeatures, usedFeaturesPerModel...)
		linregs = append(linregs, linreg...)
	}

	return linregs, usedFeatures
}
