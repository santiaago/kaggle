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
func trainModels(file string, reader Reader) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {

	linregs, featuresSpecific := trainSpecificModels(file)
	linregsByComb, featuresByComb := trainModelsByFeatureCombination(file)
	linregsWithTransform, featuresWithTransform := trainModelsWithTransform(reader)

	featuresPerModel = append(featuresPerModel, featuresSpecific...)
	featuresPerModel = append(featuresPerModel, featuresByComb...)
	featuresPerModel = append(featuresPerModel, featuresWithTransform...)

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
func trainSpecificModels(file string) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {
	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		r := csv.NewReader(csvfile)
		slrf := specificLinregFuncs()
		linregs, featuresPerModel = trainModelsByFuncs(r, slrf)
	}
	return
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(file string) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		r := csv.NewReader(csvfile)
		lrac := linregAllCombinations()
		linregsM, featuresM := trainModelsByMetaFuncs(r, lrac)

		linregs = append(linregs, linregsM...)
		featuresPerModel = append(featuresPerModel, featuresM...)
	}
	return
}

// trainModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainModelsWithTransform(reader Reader) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	d, err := reader.Read()
	if err != nil {
		log.Println("error when getting data from reader,", err)
	}

	lrWith2DTransform, ufWith2DTransform := trainModelsWith2DTransform(d)
	linregs = append(linregs, lrWith2DTransform...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, ufWith2DTransform...)

	lrWith3DTransform, ufWith3DTransform := trainModelsWith3DTransform(d)
	linregs = append(linregs, lrWith3DTransform...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, ufWith3DTransform...)

	return
}

// trainModelsWith2DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainModelsWith2DTransform(d [][]float64) ([]*linreg.LinearRegression, [][]int) {

	funcs := transform2DFuncs()
	f := passengerFeatures()
	dim := 2
	return trainModelsWithNDTransformFuncs(d, funcs, f, dim)
}

// trainModelsWith3DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainModelsWith3DTransform(d [][]float64) ([]*linreg.LinearRegression, [][]int) {

	funcs := transform3DFuncs()
	f := passengerFeatures()
	dim := 3

	return trainModelsWithNDTransformFuncs(d, funcs, f, dim)
}

// trainModelsWithNDTransformFuncs returns
//   * an array of linearRegression models
//   * an array of used features vectors
// Models are created as follows:
// Data is filtered with respect to the 'candidateFeatures' vector and the 'dimention' param.
// We generate a vector of combinations of the candidateFeatures vector.
// Each combination has the size of the size of 'dimention'.
// Each (combination, transform function) pair is a specific model.
func trainModelsWithNDTransformFuncs(d [][]float64, funcs []func([]float64) []float64, candidateFeatures []int, dimension int) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {

	combs := combinations(candidateFeatures, dimension)
	for _, comb := range combs {

		fd := filter(d, comb)

		index := 0
		for _, f := range funcs {
			if linreg, err := trainModelWithTransform(fd, f); err == nil {
				linreg.Name = fmt.Sprintf("%dD %v transformed %d", dimension, comb, index)
				fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
				linregs = append(linregs, linreg)
				featuresPerModel = append(featuresPerModel, comb)
				index++
			}
		}
	}
	return
}

// trainModelWithTransform returns a linear model and an error if it fails to learn.
// It uses the data passed as param and a transformation function.
func trainModelWithTransform(data [][]float64, f func([]float64) []float64) (*linreg.LinearRegression, error) {
	lr := linreg.NewLinearRegression()
	lr.InitializeFromData(data)
	lr.TransformFunction = f
	lr.ApplyTransformation()
	err := lr.Learn()
	return lr, err
}

// trainModelsByFuncs returns an array of linear regression models with respect
// to an array of functions passed as arguments.
// Those function takes as argument the passengers data and return a linear
// regression model.
func trainModelsByFuncs(r *csv.Reader, funcs []func(passengers []passenger) (*linreg.LinearRegression, []int)) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {

	passengers := passengersFromTrainingSet(r)
	for _, f := range funcs {
		linreg, features := f(passengers)
		linregs = append(linregs, linreg)
		featuresPerModel = append(featuresPerModel, features)
	}
	return
}

// trainModelsByMetaFuncs returns an array of linear regression models with
// respect to an array of linear regression functions passed as arguments.
// Those functions takes as argument the passengers data and
// return an array of linear regression model.
func trainModelsByMetaFuncs(r *csv.Reader, metaLinregFuncs []func(passengers []passenger) ([]*linreg.LinearRegression, [][]int)) ([]*linreg.LinearRegression, [][]int) {
	passengers := passengersFromTrainingSet(r)
	var linregs []*linreg.LinearRegression
	var features [][]int
	for _, f := range metaLinregFuncs {
		linreg, featuresPerModel := f(passengers)
		features = append(features, featuresPerModel...)
		linregs = append(linregs, linreg...)
	}
	return linregs, features
}
