package main

import (
	"fmt"

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
	model       ModelType // the model type, either logistic regression or linear regression.
	transform   Transform // the transform dimension if any.
	transformID int       // the id of the transformation function.
	features    []int     // the features to use for this model.
	regularized bool      // flag to know if model is using regularization.
	k           int       // k param used in regularization.
}

// name returns the name of the model
// a describes the model info passed in.
//
func (mi modelInfo) name() (name string) {

	if mi.model == linearRegression {
		name = "linreg"
	} else if mi.model == logisticRegression {
		name = "logreg"
	}

	if mi.transform == T3D {
		name += " 3D"
	} else if mi.transform == T4D {
		name += " 4D"
	}

	if mi.transform != NOT {
		name += fmt.Sprintf(" transformed %v", mi.transformID)
	}

	name += fmt.Sprintf(" %v", mi.features)

	if mi.regularized {
		name += fmt.Sprintf(" regularized k %v", mi.k)
	}

	return
}

// newModel creates model type with respect to the model
// info passed in.
//
func (mi modelInfo) newModel() (m ml.Model) {
	// todo(santiaago): need ml.TransformFunc type
	var transformFunc func([]float64) []float64

	if mi.transform == T3D {
		transformFunc = transform.Funcs3D()[mi.transformID]
	} else if mi.transform == T4D {
		transformFunc = transform.Funcs4D()[mi.transformID]
	}

	if mi.model == linearRegression {
		m = linreg.NewLinearRegression()
		m.(*linreg.LinearRegression).TransformFunction = transformFunc
	} else if mi.model == logisticRegression {
		m = logreg.NewLogisticRegression()
		m.(*logreg.LogisticRegression).TransformFunction = transformFunc
	}
	return
}

// model return a ml.Model with respect to the model information
// passed in.
//
func (mi modelInfo) Model(dc data.Container) *ml.Model {

	m := mi.newModel()
	if m == nil {
		return nil
	}

	fd := dc.FilterWithPredict(mi.features)

	if lr, ok := m.(*linreg.LinearRegression); ok {

		lr.InitializeFromData(fd)
		if mi.transform > NOT {
			lr.ApplyTransformation()
		}

		if mi.regularized {
			lr.K = mi.k
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
		if mi.transform > NOT {
			lr.ApplyTransformation()
		}

		if mi.regularized {
			lr.K = mi.k
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
