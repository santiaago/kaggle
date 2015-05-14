package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"

	"github.com/santiaago/kaggle/transform"
	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
	"github.com/santiaago/ml/logreg"
)

// ModelType defines the model that it is been used.
type ModelType int

const (
	linearRegression ModelType = iota
	logisticRegression
)

// Transform defines the type of transformation used.
type Transform int

const (
	NOT Transform = iota
	T3D
	T4D
)

// modelInfo is a type that describes the model to use.
//
type modelInfo struct {
	Model       ModelType // the model type, either logistic regression or linear regression.
	Transform   Transform // the transform dimension if any.
	TransformID int       // the id of the transformation function.
	Features    []int     // the features to use for this model.
	Regularized bool      // flag to know if model is using regularization.
	K           int       // k param used in regularization.
}

// ModelInfoFromModel returns a modelInfo type from
// a Model type.
//
func ModelInfoFromModel(m *ml.ModelContainer) (mi modelInfo) {

	if lr, ok := m.Model.(*linreg.LinearRegression); ok {
		mi.Model = linearRegression
		if !lr.HasTransform {
			mi.Transform = NOT
		}
		if lr.IsRegularized {
			mi.Regularized = true
			mi.K = lr.K
		}
	} else if _, ok := m.Model.(*logreg.LogisticRegression); ok {
		mi.Model = logisticRegression
		if !lr.HasTransform {
			mi.Transform = NOT
		}
		if lr.IsRegularized {
			mi.Regularized = true
			mi.K = lr.K
		}
	}
	mi.Features = m.Features
	return
}

// name returns the name of the model
// a describes the model info passed in.
//
func (mi modelInfo) name() (name string) {

	if mi.Model == linearRegression {
		name = "linreg"
	} else if mi.Model == logisticRegression {
		name = "logreg"
	}

	if mi.Transform == T3D {
		name += " 3D"
	} else if mi.Transform == T4D {
		name += " 4D"
	}

	if mi.Transform != NOT {
		name += fmt.Sprintf(" transformed %v", mi.TransformID)
	}

	name += fmt.Sprintf(" %v", mi.Features)

	if mi.Regularized {
		name += fmt.Sprintf(" regularized k %v", mi.K)
	}

	return
}

// newModel creates model type with respect to the model
// info passed in.
//
func (mi modelInfo) newModel() (m ml.Model) {
	// todo(santiaago): need ml.TransformFunc type
	var transformFunc func([]float64) ([]float64, error)

	if mi.Transform == T3D {
		transformFunc = transform.Funcs3D()[mi.TransformID]
	} else if mi.Transform == T4D {
		transformFunc = transform.Funcs4D()[mi.TransformID]
	}

	if mi.Model == linearRegression {
		m = linreg.NewLinearRegression()
		m.(*linreg.LinearRegression).TransformFunction = transformFunc
	} else if mi.Model == logisticRegression {
		m = logreg.NewLogisticRegression()
		m.(*logreg.LogisticRegression).TransformFunction = transformFunc
	}
	return
}

// model return a ml.Model with respect to the model information
// passed in.
//
func (mi modelInfo) GetModel(dc data.Container) *ml.Model {

	m := mi.newModel()
	if m == nil {
		return nil
	}

	fd := dc.FilterWithPredict(mi.Features)

	if lr, ok := m.(*linreg.LinearRegression); ok {

		lr.InitializeFromData(fd)
		if mi.Transform > NOT {
			lr.ApplyTransformation()
		}

		if mi.Regularized {
			lr.K = mi.K
			if err := lr.LearnWeightDecay(); err != nil {
				return nil
			}
			// todo(santiaago): update should be part of WeightDecay?
			lr.Wn = lr.WReg
		} else {
			if err := lr.Learn(); err != nil {
				return nil
			}
		}
	} else if lr, ok := m.(*logreg.LogisticRegression); ok {
		lr.InitializeFromData(fd)
		if mi.Transform > NOT {
			lr.ApplyTransformation()
		}

		if mi.Regularized {
			lr.K = mi.K
			if err := lr.LearnRegularized(); err != nil {
				return nil
			}
			lr.Wn = lr.WReg
		} else {
			if err := lr.Learn(); err != nil {
				return nil
			}
		}
	}
	return &m
}

func importModels(path string) (models ml.ModelContainers) {
	fmt.Printf("importing models to %v\n", path)
	var b []byte
	var err error
	if b, err = ioutil.ReadFile(path); err != nil {
		log.Printf("unable to read file %v, %v", path, err)
		return
	}

	var modelInfos []modelInfo
	if err = json.Unmarshal(b, &modelInfos); err != nil {
		log.Printf("unable to unmarshal bytes %v", err)
		return
	}

	for _, mi := range modelInfos {
		m := mi.newModel()
		mc := ml.NewModelContainer(m, mi.name(), mi.Features)
		models = append(models, mc)
	}

	fmt.Printf("Done importing %v models from %v\n", len(models), path)
	return
}

func exportModels(models ml.ModelContainers, path string) {
	fmt.Printf("exporting models to %v\n", path)

	var modelInfos []modelInfo

	for m := range models {
		if models[m] == nil {
			continue
		}

		mi := ModelInfoFromModel(models[m])
		modelInfos = append(modelInfos, mi)
	}
	var b []byte
	var err error

	if b, err = json.Marshal(modelInfos); err != nil {
		log.Printf("unable to marshal array of modelInfo objects ", err)
	}

	err = ioutil.WriteFile(path, b, 0644)
	if err != nil {
		log.Printf("unable to write to file %v, %v", path, err)
		panic(err)
	}
	fmt.Printf("Done exporting models to %v\n", path)
}
