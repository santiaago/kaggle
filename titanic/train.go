package main

import (
	"fmt"
	"log"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
	"github.com/santiaago/ml/logreg"
	"github.com/santiaago/ml/transform"
)

// trainModels returns:
// * an array of trained LinearRegression/LogisticRegression models.
// It uses the reader passed as param to read the data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// * trainModelsWithTransform
// * trainModelsWithRegularization
//
func trainModels(reader data.Reader) (models ml.ModelContainers) {
	if *verbose {
		fmt.Println("Starting training models")
	}
	var dc data.Container
	var err error
	dc, err = reader.Read()
	if err != nil {
		log.Println("error when getting the data.container from the reader,", err)
	}

	if *canImportModels {
		models = updateModels(dc, importModels(*importPath))
		return
	}

	linregModels := trainLinregModels(dc)
	models = append(models, linregModels...)

	logregModels := trainLogregModels(dc)
	models = append(models, logregModels...)
	if *verbose {
		fmt.Println("Done training models")
	}
	return
}

// trainLinregModels returns an array of modelContainers
// with the trained linear regression models specified to be trained.
// You specify which models to train by setting the trainLinreg flags.
//
func trainLinregModels(dc data.Container) (models ml.ModelContainers) {
	if !*trainLinreg {
		return
	}
	if *verbose {
		fmt.Println("training linreg models")
	}
	if *trainSpecific {
		if *verbose {
			fmt.Println("\ttraining specific")
		}
		models = append(models, specificLinregModels(dc)...)
	}

	if *combinations > 0 {
		if *verbose {
			fmt.Println("\ttraining combinations")
		}
		linregCombinationModels := trainLinregModelsByFeatureCombination(dc)
		models = append(models, linregCombinationModels...)
	}

	if *trainTransforms {
		if *verbose {
			fmt.Println("\ttraining transforms")
		}
		linregTransformModels := trainLinregModelsWithTransform(models, dc)
		models = append(models, linregTransformModels...)
	}

	if *trainRegularized {
		if *verbose {
			fmt.Println("\ttraining regularized models")
		}
		regModels := trainLinregModelsRegularized(models)
		models = append(models, regModels...)
	}
	return
}

// trainLogregModels returns an array of modelContainers
// with the trained logistic regression models specified to be trained.
// You specify which models to train by setting the trainLinreg flags.
//
func trainLogregModels(dc data.Container) (models ml.ModelContainers) {
	if !*trainLogreg {
		return
	}
	if *verbose {
		fmt.Println("training logreg models")
	}
	if *trainSpecific {
		if *verbose {
			fmt.Println("\ttraining specific")
		}
		models = append(models, specificLogregModels(dc)...)
	}

	if *combinations > 0 {
		if *verbose {
			fmt.Println("\ttraining combinations")
		}
		logregCombinationModels := trainLogregModelsByFeatureCombination(dc)
		models = append(models, logregCombinationModels...)
	}

	if *trainTransforms {
		if *verbose {
			fmt.Println("\ttraining transforms")
		}
		logregTransformModels := trainLogregModelsWithTransform(models, dc)
		models = append(models, logregTransformModels...)
	}

	if *trainRegularized {
		if *verbose {
			fmt.Println("\ttraining regularized models")
		}
		regModels := trainLogregModelsRegularized(models, dc)
		models = append(models, regModels...)
	}
	return
}

// trainLinregModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
//
func trainLinregModelsByFeatureCombination(dc data.Container) ml.ModelContainers {
	if *combinations > 0 {
		return linregCombinations(dc, *combinations)
	}
	return nil
}

// trainLogregModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
//
func trainLogregModelsByFeatureCombination(dc data.Container) ml.ModelContainers {
	if *combinations > 0 {
		return logregCombinations(dc, *combinations)
	}
	return nil
}

// trainLinregModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
//
func trainLinregModelsWithTransform(models ml.ModelContainers, dc data.Container) (transModels ml.ModelContainers) {

	return trainLinregModelsWithNDTransformFuncs(models, dc, transformArray(*transformDimension), *transformDimension)
}

// trainLogregModelsWithTransform returns:
//   * an array of linearRegression models
//   * an array of used features vectors.
//
func trainLogregModelsWithTransform(models ml.ModelContainers, dc data.Container) (transModels ml.ModelContainers) {

	return trainLogregModelsWithNDTransformFuncs(models, dc, transformArray(*transformDimension), *transformDimension)
}

// transformArray returns an array of transform functions with respect to the dimension passed in.
func transformArray(dim int) []func([]float64) ([]float64, error) {
	switch dim {
	case 2:
		return transform.Funcs2D()
	case 3:
		return transform.Funcs3D()
	case 4:
		return transform.Funcs4D()
	case 5:
		return transform.Funcs5D()
	default:
		log.Printf("transformed dimension not supported %v", *transformDimension)
		return nil
	}
}

//trainLinregModelsWithNDTransformFuncs returns
//   * an array of linearRegression models
//   * an array of used features vectors
// Models are created as follows:
// Data is filtered with respect to the 'candidateFeatures' vector and the 'dimension' param.
// We generate a vector of combinations of the candidateFeatures vector.
// Each combination has the size of the size of 'dimention'.
// Each (combination, transform function) pair is a specific model.
//
func trainLinregModelsWithNDTransformFuncs(models ml.ModelContainers, dc data.Container, funcs []func([]float64) ([]float64, error), dimension int) (transModels ml.ModelContainers) {

	for _, m := range models {
		if m == nil {
			continue
		}

		fd := dc.FilterWithPredict(m.Features)
		index := 0
		format := "linreg %dD %v transformed %d"
		for i, f := range funcs {
			if lr, err := trainLinregModelWithTransform(fd, f); err == nil {
				name := fmt.Sprintf(format, dimension, m.Features, index)
				mc := ml.NewModelContainer(lr, name, m.Features)
				mc.TransformDimension = dimension
				mc.TransformID = i
				transModels = append(transModels, mc)
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
//
func trainLogregModelsWithNDTransformFuncs(models ml.ModelContainers, dc data.Container, funcs []func([]float64) ([]float64, error), dimension int) (transModels ml.ModelContainers) {

	for _, m := range models {
		if m == nil {
			continue
		}

		fd := dc.FilterWithPredict(m.Features)
		index := 0
		format := "logreg %dD %v transformed %d epochs-%v"
		for i, f := range funcs {
			if lr, err := trainLogregModelWithTransform(fd, f); err == nil {
				name := fmt.Sprintf(format, dimension, m.Features, index, lr.Epochs)
				if *verbose {
					fmt.Printf("\r%v", name)
				}
				mc := ml.NewModelContainer(lr, name, m.Features)
				mc.TransformDimension = dimension
				mc.TransformID = i
				transModels = append(transModels, mc)
				index++
			}
		}
	}
	return
}

// trainLinregModelWithTransform returns a linear model and an error if it fails to learn.
// It uses the data passed as param and a transformation function.
// todo(santiago): export transform function type from ml.
//
func trainLinregModelWithTransform(data [][]float64, f func([]float64) ([]float64, error)) (*linreg.LinearRegression, error) {
	var err error
	lr := linreg.NewLinearRegression()
	lr.InitializeFromData(data)
	lr.TransformFunction = f
	if err = lr.ApplyTransformation(); err != nil {
		return nil, err
	}
	err = lr.Learn()
	return lr, err
}

// trainLogregModelWithTransform returns a linear model and an error if it fails to learn.
// It uses the data passed as param and a transformation function.
//
func trainLogregModelWithTransform(data [][]float64, f func([]float64) ([]float64, error)) (*logreg.LogisticRegression, error) {
	var err error
	lr := logreg.NewLogisticRegression()
	lr.InitializeFromData(data)
	lr.TransformFunction = f
	if err = lr.ApplyTransformation(); err != nil {
		return nil, err
	}

	err = lr.Learn()
	return lr, err
}

// trainLinregModelsRegularized returns an array of models that are better in sample with regularized
// option than the normal linear regression option.
// to do this we go through all trained models and try
// them with regularization if the in sample error
// is lower append it to the list of models.
//
func trainLinregModelsRegularized(models ml.ModelContainers) (regModels ml.ModelContainers) {

	for i, m := range models {
		if m == nil {
			continue
		}
		lr, ok := m.Model.(*linreg.LinearRegression)
		if !ok {
			continue
		}
		if *verbose {
			fmt.Printf("\rtraining regularized model %v %v/%v", m.Name, i, len(models))
		}
		if nlr, err := linregWithRegularization(lr); err == nil && nlr != nil {
			name := fmt.Sprintf("%v regularized k %v", m.Name, nlr.K)
			regModels = append(regModels, ml.NewModelContainer(nlr, name, m.Features))
		} else if err != nil {
			log.Printf("cannot regularized model: %v, %v", m.Name, err)
		}
	}
	fmt.Println()
	return
}

// trainLogregModelsRegularized returns the best regularized logreg model for
// each logreg model passed in.
//
func trainLogregModelsRegularized(models ml.ModelContainers, dc data.Container) (regModels ml.ModelContainers) {

	var ok bool

	for i, m := range models {
		var lr *logreg.LogisticRegression
		if lr, ok = m.Model.(*logreg.LogisticRegression); !ok {
			continue
		}
		if lr.IsRegularized { // skip models that are already regularized<
			continue
		}
		if *verbose {
			fmt.Printf("\rtraining regularized model %v %v/%v", m.Name, i, len(models))
		}
		fd := dc.FilterWithPredict(m.Features)

		for k := -5; k < 5; k++ {
			if *verbose {
				fmt.Printf("\rregularizing model %v/%v with k:%v", i, len(models), k)
			}
			var nlr *logreg.LogisticRegression
			if nlr = logregFromK(k, fd, lr); nlr == nil {
				continue
			}

			// todo(santiaago) clean this up
			nlr.Wn = nlr.WReg
			nlr.IsRegularized = true
			name := fmt.Sprintf("%v regularized k %v", m.Name, k)
			name += fmt.Sprintf(" epochs %v", nlr.Epochs)

			mc := ml.NewModelContainer(nlr, name, m.Features)
			regModels = append(regModels, mc)
		}
	}
	fmt.Println()
	return
}

// logregFromK return a logistic regression model
// based on the k parameter passed in and the logreg passed in.
//
func logregFromK(k int, fd [][]float64, lr *logreg.LogisticRegression) *logreg.LogisticRegression {
	nlr := logreg.NewLogisticRegression()
	nlr.InitializeFromData(fd)

	if lr.HasTransform {
		nlr.TransformFunction = lr.TransformFunction
		nlr.ApplyTransformation()
	}

	nlr.K = k
	if err := nlr.LearnRegularized(); err != nil {
		log.Printf("error calling logreg.LearnRegularized, %v\n", err)
		return nil
	}
	return nlr
}

// updateModels re-trains all the models passed in.
//
func updateModels(dc data.Container, models ml.ModelContainers) (trainedModels ml.ModelContainers) {

	for _, mc := range models {
		if lr, ok := mc.Model.(*linreg.LinearRegression); ok {
			fd := dc.FilterWithPredict(mc.Features)
			lr.InitializeFromData(fd)
			if lr.HasTransform {
				lr.TransformFunction = transformArray(mc.TransformDimension)[mc.TransformID]
				lr.ApplyTransformation()
			}
			if !lr.IsRegularized {
				if err := lr.Learn(); err != nil {
					log.Printf("unable to train model %v\n", mc.Name)
					continue
				}
				trainedModels = append(trainedModels, mc)
			}
		} else if lr, ok := mc.Model.(*logreg.LogisticRegression); ok {
			fd := dc.FilterWithPredict(mc.Features)
			lr.InitializeFromData(fd)
			if lr.HasTransform {
				lr.TransformFunction = transformArray(mc.TransformDimension)[mc.TransformID]
				lr.ApplyTransformation()
			}
			if !lr.IsRegularized {
				if err := lr.Learn(); err != nil {
					log.Printf("unable to train model %v\n", mc.Name)
					continue
				}
				trainedModels = append(trainedModels, mc)
			}
		}
	}
	return
}
