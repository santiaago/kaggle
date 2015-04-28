package main

import (
	"fmt"
	"log"

	"github.com/santiaago/kaggle/itertools"
	"github.com/santiaago/kaggle/transform"
	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
)

// trainModels returns:
// * an array of trained LinearRegression models.
// It uses the reader passed as param to read the data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// * trainModelsWithTransform
func trainModels(reader data.Reader) (models ml.ModelContainers) {

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

	regModels := trainModelsRegularized(models)
	models = append(models, regModels...)

	return
}

// trainModelsRegularized returns an array of models that are better with regularized
// option than the normal linear regression option.
// to do this we go through all trained models and try
// them with regularization if the in sample error
// is lower append it to the list of models.
func trainModelsRegularized(models ml.ModelContainers) ml.ModelContainers {
	var rModels ml.ModelContainers
	for _, m := range models {
		if m == nil {
			continue
		}
		if nlr, err := linregWithRegularization(m.Model.(*linreg.LinearRegression)); err == nil && nlr != nil {
			name := fmt.Sprintf("%v regularized k %v", m.Name, nlr.K)
			rModels = append(rModels, ml.NewModelContainer(nlr, name, m.Features))
		}
	}
	return rModels
}

// trainSpecificModels trains the following models:
// * linregSexAge
// * linregPClassAge
// * linregPClassSex
// * linregSexAgePClass
// It returns an array of all the linear regression models trained.
func trainSpecificModels(dc data.Container) ml.ModelContainers {

	modelFuncs := specificLinregFuncs()
	return ml.ModelsFromFuncs(dc, modelFuncs)
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(dc data.Container) ml.ModelContainers {

	modelFuncs := linregAllCombinations()
	return ml.ModelsFromMetaFuncs(dc, modelFuncs)
}

// trainModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainModelsWithTransform(dc data.Container) ml.ModelContainers {

	var models ml.ModelContainers

	models2D := trainModelsWith2DTransform(dc)
	models = append(models, models2D...)

	models3D := trainModelsWith3DTransform(dc)
	models = append(models, models3D...)

	models4D := trainModelsWith4DTransform(dc)
	models = append(models, models4D...)

	return models
}

// trainModelsWith2DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainModelsWith2DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs2D()
	dim := 2
	return trainModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainModelsWith3DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainModelsWith3DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs3D()
	dim := 3
	return trainModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainModelsWith4DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 4D transformation functions.
func trainModelsWith4DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs4D()
	dim := 4
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
func trainModelsWithNDTransformFuncs(dc data.Container, funcs []func([]float64) []float64, dimension int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, dimension)
	for _, c := range combs {

		fd := dc.FilterWithPredict(c)
		index := 0
		for _, f := range funcs {
			if lr, err := trainModelWithTransform(fd, f); err == nil {
				name := fmt.Sprintf("%dD %v transformed %d", dimension, c, index)
				models = append(models, ml.NewModelContainer(lr, name, c))
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
