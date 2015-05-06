package main

import (
	"fmt"
	"log"

	"github.com/santiaago/kaggle/itertools"
	"github.com/santiaago/kaggle/transform"
	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
	"github.com/santiaago/ml/logreg"
)

type ModelType int

const (
	linearRegression ModelType = iota
	logisticRegression
)

type Transform int

const (
	NOT Transform = iota
	T3D
	T4D
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

	// 1 - use ranking to generate models.
	rankedModels := []struct {
		model       ModelType
		transform   Transform
		transformID int
		features    []int
		regularized bool
		k           int
	}{
		{linearRegression, T4D, 4, []int{2, 4, 7, 11}, true, 2},
		{linearRegression, T3D, 4, []int{2, 4, 11}, false, 0},
		{logisticRegression, T3D, 1, []int{2, 4, 11}, false, 0},
		{logisticRegression, T3D, 4, []int{2, 4, 11}, false, 0},
		{linearRegression, T4D, 6, []int{2, 4, 6, 7}, true, 2},
	}

	for _, modelInfo := range rankedModels {
		var m ml.Model
		var name string
		if modelInfo.model == linearRegression {
			m = linreg.NewLinearRegression()
			name = "linreg"
			if modelInfo.transform == T3D {
				name += " 3D"
				switch modelInfo.transformID {
				case 1:
					m.(*linreg.LinearRegression).TransformFunction = transform.Funcs3D()[1]
					name += " transformed 1"
				case 4:
					m.(*linreg.LinearRegression).TransformFunction = transform.Funcs3D()[4]
					name += " transformed 4"
				}
			} else if modelInfo.transform == T4D {
				name += " 4D"
				switch modelInfo.transformID {
				case 1:
					m.(*linreg.LinearRegression).TransformFunction = transform.Funcs4D()[1]
					name += " transformed 1"
				case 4:
					m.(*linreg.LinearRegression).TransformFunction = transform.Funcs4D()[4]
					name += " transformed 4"
				case 6:
					m.(*linreg.LinearRegression).TransformFunction = transform.Funcs4D()[6]
					name += " transformed 6"
				}

			}
		} else if modelInfo.model == logisticRegression {
			name = "logreg"
			m = logreg.NewLogisticRegression()
			if modelInfo.transform == T3D {
				name += " 3D"
				switch modelInfo.transformID {
				case 1:
					m.(*logreg.LogisticRegression).TransformFunction = transform.Funcs3D()[1]
					name += " transformed 1"
				case 4:
					m.(*logreg.LogisticRegression).TransformFunction = transform.Funcs3D()[4]
					name += " transformed 4"
				}
			} else if modelInfo.transform == T4D {
				name += " 4D"
				switch modelInfo.transformID {
				case 1:
					m.(*logreg.LogisticRegression).TransformFunction = transform.Funcs4D()[1]
					name += " transformed 1"
				case 4:
					m.(*logreg.LogisticRegression).TransformFunction = transform.Funcs4D()[4]
					name += " transformed 4"
				case 6:
					m.(*logreg.LogisticRegression).TransformFunction = transform.Funcs4D()[6]
					name += " transformed 6"
				}
			}
		}

		if m != nil {
			var mc *ml.ModelContainer
			if modelInfo.model == linearRegression {
				if modelInfo.regularized {
					lr := m.(*linreg.LinearRegression)
					fd := dc.FilterWithPredict(modelInfo.features)
					lr.InitializeFromData(fd)
					lr.K = modelInfo.k
					name += fmt.Sprintf(" %v regularized k %v", modelInfo.features, modelInfo.k)
					if err := lr.LearnWeightDecay(); err != nil {
						continue
					}
					mc = ml.NewModelContainer(lr, name, modelInfo.features)
				} else {
					lr := m.(*linreg.LinearRegression)
					fd := dc.FilterWithPredict(modelInfo.features)
					lr.InitializeFromData(fd)
					name += fmt.Sprintf(" %v", modelInfo.features)
					if err := lr.Learn(); err != nil {
						continue
					}
					mc = ml.NewModelContainer(lr, name, modelInfo.features)
				}
			} else if modelInfo.model == logisticRegression {
				if modelInfo.regularized {
					lr := m.(*logreg.LogisticRegression)
					fd := dc.FilterWithPredict(modelInfo.features)
					lr.InitializeFromData(fd)
					lr.K = modelInfo.k
					name += fmt.Sprintf(" %v regularized k %v", modelInfo.features, modelInfo.k)
					if err := lr.LearnRegularized(); err != nil {
						continue
					}
					mc = ml.NewModelContainer(lr, name, modelInfo.features)
				} else {
					lr := m.(*logreg.LogisticRegression)
					fd := dc.FilterWithPredict(modelInfo.features)
					lr.InitializeFromData(fd)
					name += fmt.Sprintf(" %v", modelInfo.features)
					if err := lr.Learn(); err != nil {
						continue
					}
					mc = ml.NewModelContainer(lr, name, modelInfo.features)
				}
			}
			models = append(models, mc)

		}
	}
	// var nmodels ml.ModelContainers
	// fmt.Println(len(models))
	// for _, m := range models {
	// 	if lr, ok := m.Model.(*logreg.LogisticRegression); ok {
	// 		for k := -50; k < 50; k++ {
	// 			nlr := logreg.NewLogisticRegression()
	// 			fd := dc.FilterWithPredict(m.Features)
	// 			nlr.InitializeFromData(fd)
	// 			nlr.TransformFunction = lr.TransformFunction
	// 			nlr.K = k
	// 			name := fmt.Sprintf("%v regularized k %v", m.Name, k)
	// 			if err := nlr.LearnRegularized(); err != nil {
	// 				log.Println("error calling logreg.LearnRegularized, %v", err)
	// 				continue
	// 			}
	// 			name += fmt.Sprintf(" epochs %v", nlr.Epochs)
	// 			nmodels = append(nmodels, ml.NewModelContainer(nlr, name, m.Features))
	// 		}
	// 	}
	// }
	// models = append(models, nmodels...)
	// 2 - generate all modes.

	// specificModels := trainSpecificModels(dc)
	// models = append(models, specificModels...)

	// linregCombinationModels := trainLinregModelsByFeatureCombination(dc)
	// models = append(models, linregCombinationModels...)

	// linregTransformModels := trainLinregModelsWithTransform(dc)
	// models = append(models, linregTransformModels...)

	// regModels := trainModelsRegularized(models)
	// models = append(models, regModels...)

	// logregModels := trainLogregSpecificModels(dc)
	// models = append(models, logregModels...)

	// logregCombinationModels := trainLogregModelsByFeatureCombination(dc)
	// models = append(models, logregCombinationModels...)

	// logregTransformModels := trainLogregModelsWithTransform(dc)
	// models = append(models, logregTransformModels...)

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
		lr, ok := m.Model.(*linreg.LinearRegression)
		if !ok {
			continue
		}
		if nlr, err := linregWithRegularization(lr); err == nil && nlr != nil {
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

func trainLogregSpecificModels(dc data.Container) ml.ModelContainers {
	modelFuncs := specificLogregFuncs()
	return ml.ModelsFromFuncs(dc, modelFuncs)
}

// trainLinregModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainLinregModelsByFeatureCombination(dc data.Container) ml.ModelContainers {

	modelFuncs := linregAllCombinations()
	return ml.ModelsFromMetaFuncs(dc, modelFuncs)
}

// trainLogregModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainLogregModelsByFeatureCombination(dc data.Container) ml.ModelContainers {

	modelFuncs := logregAllCombinations()
	return ml.ModelsFromMetaFuncs(dc, modelFuncs)
}

// trainLinregModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainLinregModelsWithTransform(dc data.Container) ml.ModelContainers {

	var models ml.ModelContainers

	// models2D := trainLinregModelsWith2DTransform(dc)
	// models = append(models, models2D...)

	models3D := trainLinregModelsWith3DTransform(dc)
	models = append(models, models3D...)

	models4D := trainLinregModelsWith4DTransform(dc)
	models = append(models, models4D...)

	return models
}

// trainLogregModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
func trainLogregModelsWithTransform(dc data.Container) ml.ModelContainers {

	var models ml.ModelContainers

	// models2D := trainLogregModelsWith2DTransform(dc)
	// models = append(models, models2D...)

	models3D := trainLogregModelsWith3DTransform(dc)
	models = append(models, models3D...)

	models4D := trainLogregModelsWith4DTransform(dc)
	models = append(models, models4D...)

	return models
}

// trainLinregModelsWith2DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainLinregModelsWith2DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs2D()
	dim := 2
	return trainLinregModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainLinregModelsWith3DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainLinregModelsWith3DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs3D()
	dim := 3
	return trainLinregModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainLinregModelsWith4DTransform returns a list of linear regression models and the corresponding feature used.
// models learn based on some 4D transformation functions.
func trainLinregModelsWith4DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs4D()
	dim := 4
	return trainLinregModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainLogregModelsWith2DTransform returns a list of logistic regression models and the corresponding feature used.
// models learn based on some 2D transformation functions.
func trainLogregModelsWith2DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs2D()
	dim := 2
	return trainLogregModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainLogregModelsWith3DTransform returns a list of logistic regression models and the corresponding feature used.
// models learn based on some 3D transformation functions.
func trainLogregModelsWith3DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs3D()
	dim := 3
	return trainLogregModelsWithNDTransformFuncs(dc, funcs, dim)
}

// trainLogregModelsWith4DTransform returns a list of logistic regression models and the corresponding feature used.
// models learn based on some 4D transformation functions.
func trainLogregModelsWith4DTransform(dc data.Container) ml.ModelContainers {

	funcs := transform.Funcs4D()
	dim := 4
	return trainLogregModelsWithNDTransformFuncs(dc, funcs, dim)
}

//trainLinregModelsWithNDTransformFuncs returns
//   * an array of linearRegression models
//   * an array of used features vectors
// Models are created as follows:
// Data is filtered with respect to the 'candidateFeatures' vector and the 'dimension' param.
// We generate a vector of combinations of the candidateFeatures vector.
// Each combination has the size of the size of 'dimention'.
// Each (combination, transform function) pair is a specific model.
func trainLinregModelsWithNDTransformFuncs(dc data.Container, funcs []func([]float64) []float64, dimension int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, dimension)
	for _, c := range combs {

		fd := dc.FilterWithPredict(c)
		index := 0
		for _, f := range funcs {
			if lr, err := trainLinregModelWithTransform(fd, f); err == nil {
				name := fmt.Sprintf("linreg %dD %v transformed %d", dimension, c, index)
				models = append(models, ml.NewModelContainer(lr, name, c))
				index++
			}
		}
	}
	return
}

//trainLogregModelsWithNDTransformFuncs returns
//   * an array of linearRegression models
//   * an array of used features vectors
// Models are created as follows:
// Data is filtered with respect to the 'candidateFeatures' vector and the 'dimension' param.
// We generate a vector of combinations of the candidateFeatures vector.
// Each combination has the size of the size of 'dimention'.
// Each (combination, transform function) pair is a specific model.
func trainLogregModelsWithNDTransformFuncs(dc data.Container, funcs []func([]float64) []float64, dimension int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, dimension)
	for _, c := range combs {

		fd := dc.FilterWithPredict(c)
		index := 0
		for _, f := range funcs {
			if lr, err := trainLogregModelWithTransform(fd, f); err == nil {
				name := fmt.Sprintf("logreg %dD %v transformed %d epochs-%v", dimension, c, index, lr.Epochs)
				fmt.Println(name)
				models = append(models, ml.NewModelContainer(lr, name, c))
				index++
			}
		}
	}
	return
}

// trainLinregModelWithTransform returns a linear model and an error if it fails to learn.
// It uses the data passed as param and a transformation function.
func trainLinregModelWithTransform(data [][]float64, f func([]float64) []float64) (*linreg.LinearRegression, error) {
	lr := linreg.NewLinearRegression()
	lr.InitializeFromData(data)
	lr.TransformFunction = f
	lr.ApplyTransformation()
	err := lr.Learn()
	return lr, err
}

// trainLogregModelWithTransform returns a linear model and an error if it fails to learn.
// It uses the data passed as param and a transformation function.
func trainLogregModelWithTransform(data [][]float64, f func([]float64) []float64) (*logreg.LogisticRegression, error) {
	lr := logreg.NewLogisticRegression()
	lr.InitializeFromData(data)
	lr.TransformFunction = f
	lr.ApplyTransformation()
	err := lr.Learn()
	return lr, err
}
