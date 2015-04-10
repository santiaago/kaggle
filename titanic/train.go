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
func trainModels(file string) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {

	linregs, featuresSpecific := trainSpecificModels(file)
	linregsByComb, featuresByComb := trainModelsByFeatureCombination(file)
	linregsWithTransform, featuresWithTransform := trainModelsWithTransform(file)

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
func trainSpecificModels(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {
	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		linregs, usedFeaturesPerModel = trainModelsByFuncs(csv.NewReader(csvfile), specificLinregFuncs())
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
		linregsOf, usedFeaturesOf := trainModelsByMetaFuncs(csv.NewReader(csvfile), linregVectorsOfInterval())

		linregs = append(linregs, linregsOf...)
		usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesOf...)
	}
	return
}

// trainModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainModelsWithTransform(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	lrWith2DTransform, ufWith2DTransform := trainModelsWith2DTransform(file)
	linregs = append(linregs, lrWith2DTransform...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, ufWith2DTransform...)

	lrWith3DTransform, ufWith3DTransform := trainModelsWith3DTransform(file)
	linregs = append(linregs, lrWith3DTransform...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, ufWith3DTransform...)

	return
}

// trainModelsWith2DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainModelsWith2DTransform(file string) (linregs []*linreg.LinearRegression, features [][]int) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		r := csv.NewReader(csvfile)
		funcs := transform2DFuncs()
		f := passengerFeatures()
		d := 2
		linregs, features = trainModelsWithNDTransformFuncs(r, funcs, f, d)
	}
	return
}

// trainModelsWith3DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainModelsWith3DTransform(file string) (linregs []*linreg.LinearRegression, features [][]int) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		r := csv.NewReader(csvfile)
		funcs := transform3DFuncs()
		f := passengerFeatures()
		d := 3
		linregs, features = trainModelsWithNDTransformFuncs(r, funcs, f, d)
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
func trainModelsWithNDTransformFuncs(r *csv.Reader, funcs []func([]float64) []float64, candidateFeatures []int, dimension int) (linregs []*linreg.LinearRegression, featuresPerModel [][]int) {

	// todo(santiaago): this could be changed to
	// passengerExtractor implements an extract method
	// to read data from a csv.Reader and generate an
	// array of passengers.
	// something like.
	// passengers := passengerExtractor.Extract(r)
	passengers := passengersFromTrainingSet(r)

	// todo:(santiaago): this could be changed to
	// passengerCleaner that implement a clean method
	// to modify a passenger array into more useful data for a
	// linreg model.
	data := prepareData(passengers)

	combs := combinations(candidateFeatures, dimension)
	for _, comb := range combs {

		filteredData := filter(data, comb)

		index := 0
		for _, f := range funcs {
			if linreg, err := trainModelWithTransform(filteredData, f); err == nil {
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
