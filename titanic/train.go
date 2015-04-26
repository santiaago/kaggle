package main

import (
	"fmt"
	"log"

	"github.com/santiaago/kaggle/data"
	"github.com/santiaago/kaggle/itertools"
	"github.com/santiaago/kaggle/transform"
	"github.com/santiaago/ml/linreg"
)

// trainModels returns:
// * an array of trained LinearRegression models.
// It uses the reader passed as param to read the data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// * trainModelsWithTransform
func trainModels(reader data.Reader) (models []*modelContainer) {

	var dc data.Container
	var err error
	dc, err = reader.Read()
	if err != nil {
		log.Println("error when getting the data.container from the reader,", err)
	}

	specificModels := trainSpecificModels(dc)
	models = append(models, specificModels...)

	combinationModels := trainModelsByFeatureCombination(dc)
	models = append(models, combinationModels...)

	transformModels := trainModelsWithTransform(dc)
	models = append(models, transformModels...)

	return
}

// trainSpecificModels trains the following models:
// * linregSexAge
// * linregPClassAge
// * linregPClassSex
// * linregSexAgePClass
// It returns an array of all the linear regression models trained.
func trainSpecificModels(dc data.Container) []*modelContainer {

	models := specificLinregFuncs()
	return trainModelsByFuncs(dc, models)
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(dc data.Container) []*modelContainer {

	models := linregAllCombinations()
	return trainModelsByMetaFuncs(dc, models)
}

// trainModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainModelsWithTransform(dc data.Container) []*modelContainer {

	var models []*modelContainer

	models2D := trainModelsWith2DTransform(dc)
	models = append(models, models2D...)

	models3D := trainModelsWith3DTransform(dc)
	models = append(models, models3D...)

	return models
}

// trainModelsWith2DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainModelsWith2DTransform(dc data.Container) []*modelContainer {

	funcs := transform.Funcs2D()
	dim := 2
	return trainModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainModelsWith3DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainModelsWith3DTransform(dc data.Container) []*modelContainer {

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
func trainModelsWithNDTransformFuncs(dc data.Container, funcs []func([]float64) []float64, dimension int) (models []*modelContainer) {

	combs := itertools.Combinations(dc.Features, dimension)
	for _, c := range combs {

		fd := filter(dc.Data, c)
		index := 0
		for _, f := range funcs {
			if lr, err := trainModelWithTransform(fd, f); err == nil {
				name := fmt.Sprintf("%dD %v transformed %d", dimension, c, index)
				models = append(models, NewModelContainer(lr, name, c))
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
func trainModelsByFuncs(dc data.Container, funcs []func(data.Container) (*modelContainer, error)) (models []*modelContainer) {

	for _, f := range funcs {
		if m, err := f(dc); err == nil {
			models = append(models, m)
		}
	}
	return
}

// trainModelsByMetaFuncs returns an array of linear regression models with
// respect to an array of linear regression functions passed as arguments.
// Those functions takes as argument a 2 dimensional array of data and
// return an array of linear regression model.
// todo(santiaago): rename func and params
func trainModelsByMetaFuncs(dc data.Container, metaLinregFuncs []func(data.Container) []*modelContainer) []*modelContainer {

	var allModels []*modelContainer
	for _, f := range metaLinregFuncs {
		models := f(dc)
		allModels = append(allModels, models...)
	}
	return allModels
}
