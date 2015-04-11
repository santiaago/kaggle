package main

import (
	"fmt"
	"log"

	"github.com/santiaago/caltechx.go/linreg"
	"github.com/santiaago/kaggle/data"
	"github.com/santiaago/kaggle/itertools"
	"github.com/santiaago/kaggle/transform"
)

// trainModels returns:
// * an array of trained LinearRegression models.
// It uses the reader passed as param to read the data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// * trainModelsWithTransform
func trainModels(reader data.Reader) (linregs regressions, featuresPerModel [][]int) {
	var dc data.Container
	var err error
	dc, err = reader.Read()
	if err != nil {
		log.Println("error when getting the data.container from the reader,", err)
	}

	linregs, featuresSpecific := trainSpecificModels(dc)
	linregsByComb, featuresByComb := trainModelsByFeatureCombination(dc)
	linregsWithTransform, featuresWithTransform := trainModelsWithTransform(dc)

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
func trainSpecificModels(dc data.Container) (linregs regressions, featuresPerModel [][]int) {

	slrf := specificLinregFuncs()
	linregs, featuresPerModel = trainModelsByFuncs(dc, slrf)
	return
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(dc data.Container) (linregs regressions, featuresPerModel [][]int) {

	lrac := linregAllCombinations()
	linregsM, featuresM := trainModelsByMetaFuncs(dc, lrac)

	linregs = append(linregs, linregsM...)
	featuresPerModel = append(featuresPerModel, featuresM...)
	return
}

// trainModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainModelsWithTransform(dc data.Container) (linregs regressions, usedFeaturesPerModel [][]int) {

	lrWith2DTransform, ufWith2DTransform := trainModelsWith2DTransform(dc)
	linregs = append(linregs, lrWith2DTransform...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, ufWith2DTransform...)

	lrWith3DTransform, ufWith3DTransform := trainModelsWith3DTransform(dc)
	linregs = append(linregs, lrWith3DTransform...)
	usedFeaturesPerModel = append(usedFeaturesPerModel, ufWith3DTransform...)

	return
}

// trainModelsWith2DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainModelsWith2DTransform(dc data.Container) (regressions, [][]int) {

	funcs := transform.Funcs2D()
	dim := 2
	return trainModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainModelsWith3DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainModelsWith3DTransform(dc data.Container) (regressions, [][]int) {

	funcs := transform.Funcs3D()
	dim := 3

	return trainModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainModelsWithNDTransformFuncs returns
//   * an array of linearRegression models
//   * an array of used features vectors
// Models are created as follows:
// Data is filtered with respect to the 'candidateFeatures' vector and the 'dimension' param.
// We generate a vector of combinations of the candidateFeatures vector.
// Each combination has the size of the size of 'dimention'.
// Each (combination, transform function) pair is a specific model.
func trainModelsWithNDTransformFuncs(dc data.Container, funcs []func([]float64) []float64, dimension int) (linregs regressions, featuresPerModel [][]int) {

	combs := itertools.Combinations(dc.Features, dimension)
	for _, comb := range combs {

		fd := filter(dc.Data, comb)
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
// Those function takes as argument a 2 dimensional data array and return a linear
// regression model.
func trainModelsByFuncs(dc data.Container, funcs []func(data.Container) (*linreg.LinearRegression, []int)) (linregs regressions, featuresPerModel [][]int) {

	for _, f := range funcs {
		linreg, features := f(dc)
		linregs = append(linregs, linreg)
		featuresPerModel = append(featuresPerModel, features)
	}
	return
}

// trainModelsByMetaFuncs returns an array of linear regression models with
// respect to an array of linear regression functions passed as arguments.
// Those functions takes as argument a 2 dimensional array of data and
// return an array of linear regression model.
func trainModelsByMetaFuncs(dc data.Container, metaLinregFuncs []func(data.Container) (regressions, [][]int)) (regressions, [][]int) {
	var linregs regressions
	var features [][]int
	for _, f := range metaLinregFuncs {
		linreg, featuresPerModel := f(dc)
		features = append(features, featuresPerModel...)
		linregs = append(linregs, linreg...)
	}
	return linregs, features
}
