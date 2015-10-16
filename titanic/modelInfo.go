package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
	"github.com/santiaago/ml/logreg"
	"github.com/santiaago/ml/svm"
	"github.com/santiaago/ml/transform"
)

// ModelType defines the model that it is been used.
type ModelType int

const (
	linearRegression ModelType = iota
	logisticRegression
	supportVectorMachines
)

// Dimension defines the type of transformation used.
type Dimension int

const (
	NOT Dimension = 0
	T3D Dimension = 3
	T4D Dimension = 4
	T5D Dimension = 5
)

// modelInfo is a type that describes the model to use.
//
type modelInfo struct {
	Model              ModelType // the model type, either logistic regression or linear regression.
	TransformDimension Dimension // the transform dimension if any.
	TransformID        int       // the id of the transformation function.
	Features           []int     // the features to use for this model.
	Regularized        bool      // flag to know if model is using regularization.
	K                  int       // k param used in regularization or in svm.
	T                  int       // param used in svm algorithm.
	L                  float64   // param used in svm algorithm.
}

// ModelInfoFromModel returns a modelInfo type from
// a Model type.
//
func ModelInfoFromModel(m *ml.ModelContainer) (mi modelInfo) {

	if lr, ok := m.Model.(*linreg.LinearRegression); ok {
		mi.Model = linearRegression
		if !lr.HasTransform {
			mi.TransformDimension = NOT
		}
		mi.TransformDimension = Dimension(m.TransformDimension)
		mi.TransformID = m.TransformID
		if lr.IsRegularized {
			mi.Regularized = true
			mi.K = lr.K
		}
	} else if lr, ok := m.Model.(*logreg.LogisticRegression); ok {
		mi.Model = logisticRegression
		if !lr.HasTransform {
			mi.TransformDimension = NOT
		}
		mi.TransformDimension = Dimension(m.TransformDimension)
		mi.TransformID = m.TransformID
		if lr.IsRegularized {
			mi.Regularized = true
			mi.K = lr.K
		}
	} else if svm, ok := m.Model.(*svm.SVM); ok {
		mi.Model = supportVectorMachines
		if !svm.HasTransform {
			mi.TransformDimension = NOT
		}
		mi.TransformDimension = Dimension(m.TransformDimension)
		mi.TransformID = m.TransformID
		mi.K = svm.K
		mi.T = svm.T
		mi.L = svm.Lambda
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
	} else if mi.Model == supportVectorMachines {
		name = "svm"
	}

	if mi.TransformDimension == T3D {
		name += " 3D"
	} else if mi.TransformDimension == T4D {
		name += " 4D"
	} else if mi.TransformDimension == T5D {
		name += " 5D"
	}

	name += fmt.Sprintf(" %v", mi.Features)

	if mi.Model == supportVectorMachines {
		k := mi.K
		t := mi.T
		l := mi.L
		if *svmKOverride {
			k = *svmK
		}
		if *svmTOverride {
			t = *svmT
		}
		if *svmLambdaOverride {
			l = *svmLambda
		}
		name += fmt.Sprintf(" k %v T %v L %v", k, t, l)
	}

	if mi.TransformDimension != NOT {
		name += fmt.Sprintf(" transformed %v", mi.TransformID)
	}

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

	if mi.TransformDimension == T3D {
		transformFunc = transform.Funcs3D()[mi.TransformID]
	} else if mi.TransformDimension == T4D {
		transformFunc = transform.Funcs4D()[mi.TransformID]
	} else if mi.TransformDimension == T5D {
		transformFunc = transform.Funcs5D()[mi.TransformID]
	}

	if mi.Model == linearRegression {
		m = linreg.NewLinearRegression()
		m.(*linreg.LinearRegression).TransformFunction = transformFunc
		if mi.TransformDimension > 0 {
			m.(*linreg.LinearRegression).HasTransform = true
		}
	} else if mi.Model == logisticRegression {
		m = logreg.NewLogisticRegression()
		m.(*logreg.LogisticRegression).TransformFunction = transformFunc
		if mi.TransformDimension > 0 {
			m.(*logreg.LogisticRegression).HasTransform = true
		}
	} else if mi.Model == supportVectorMachines {
		m = svm.NewSVM()
		m.(*svm.SVM).TransformFunction = transformFunc
		if mi.TransformDimension > 0 {
			m.(*svm.SVM).HasTransform = true
		}
		m.(*svm.SVM).K = mi.K
		m.(*svm.SVM).T = mi.T
		m.(*svm.SVM).Lambda = mi.L
		if *verbose {
			fmt.Printf("setting svm with k: %v T: %v L: %v\n", mi.K, mi.T, mi.L)
		}
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
		if mi.TransformDimension > NOT {
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
		if mi.TransformDimension > NOT {
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
	} else if svm, ok := m.(*svm.SVM); ok {
		svm.InitializeFromData(fd)
		if mi.TransformDimension > NOT {
			svm.ApplyTransformation()
		}
		if err := lr.Learn(); err != nil {
			return nil
		}
	}
	return &m
}

func importModels(path string) (models ml.ModelContainers) {
	if *verbose {
		fmt.Printf("importing models from %v\n", path)
	}
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
		mc.TransformDimension = int(mi.TransformDimension)
		mc.TransformID = mi.TransformID
		models = append(models, mc)
	}
	if *verbose {
		fmt.Printf("Done importing %v models from %v\n", len(models), path)
	}
	return
}

func exportModels(models ml.ModelContainers, path string) {
	if !*canExportModels {
		return
	}

	if *verbose {
		fmt.Printf("exporting models to %v\n", path)
	}

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

	if b, err = json.MarshalIndent(modelInfos, "", "    "); err != nil {
		log.Printf("unable to marshal array of modelInfo objects ", err)
	}

	err = ioutil.WriteFile(path, b, 0644)
	if err != nil {
		log.Printf("unable to write to file %v, %v", path, err)
		panic(err)
	}
	if *verbose {
		fmt.Printf("Done exporting models to %v\n", path)
	}
}
