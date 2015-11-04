package main

import (
	"testing"

	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/svm"
)

func TestSVMImportModels(t *testing.T) {

	msvm := svm.NewSVM()
	msvm.K = 20
	msvm.Lambda = 0.001
	msvm.T = 1000
	models := importModels("svm.json")
	isvm := models[0].Model.(*svm.SVM)
	checkSVMs(t, msvm, isvm)
}

func TestSVMAreConsistent(t *testing.T) {

	var dc data.Container
	var err error
	if dc, err = buildContainer(); err != nil {
		t.Error(err)
	}

	svm1 := createSVM(t, dc)
	svm2 := createSVM(t, dc)

	checkSVMs(t, svm1, svm2)
}

func TestSVMImportLearn(t *testing.T) {

	var dc data.Container
	var err error
	if dc, err = buildContainer(); err != nil {
		t.Error(err)
	}

	svm := createSVM(t, dc)
	isvm := importSVM(t, dc)

	checkSVMs(t, svm, isvm)
}

func buildContainer() (data.Container, error) {
	reader := NewPassengerReader("data/train.csv", NewPassengerTrainExtractor())
	return reader.Read()
}

func createSVM(t *testing.T, dc data.Container) *svm.SVM {
	svm := svm.NewSVM()
	svm.K = 20
	svm.Lambda = 0.001
	svm.T = 1000

	features := []int{2, 4, 6, 7, 8, 9, 10}
	fd := dc.FilterWithPredict(features)
	svm.InitializeFromData(fd)
	if err := svm.Learn(); err != nil {
		t.Error(err)
	}
	return svm
}

func importSVM(t *testing.T, dc data.Container) *svm.SVM {
	models := importModels("svm.json")
	svm := models[0].Model.(*svm.SVM)
	fd := dc.FilterWithPredict(models[0].Features)
	svm.InitializeFromData(fd)
	if err := svm.Learn(); err != nil {
		t.Error(err)
	}
	return svm
}

// checkSVMs checks that fields of SVMs are equal
// except for Wn that is computed through the Pegasos algorithm that
// involves random selection of indexes.
//
func checkSVMs(t *testing.T, a, b *svm.SVM) {
	if a.K != b.K {
		t.Errorf("K param are different svmA.K:%v svmB.K:%v", a.K, b.K)
	} else if a.T != b.T {
		t.Errorf("T param are different svmA.T:%v svmB.T:%v", a.T, b.T)
	} else if a.Eta != b.Eta {
		t.Errorf("Eta param are different svmA.Eta:%v svmB.Eta:%v", a.Eta, b.Eta)
	} else if a.VectorSize != b.VectorSize {
		t.Errorf("VectorSize param are different svmA.VectorSize:%v svmB.VectorSize:%v", a.VectorSize, b.VectorSize)
	} else if a.TrainingPoints != b.TrainingPoints {
		t.Errorf("TrainingPoints param are different svmA.TrainingPoints:%v svmB.TrainingPoints:%v", a.TrainingPoints, b.TrainingPoints)
	} else if a.HasTransform != b.HasTransform {
		t.Errorf("HasTransform param are different svmA.HasTransform:%v svmB.HasTransform:%v", a.HasTransform, b.HasTransform)
	} else if !equal(a.Yn, b.Yn) {
		t.Errorf("svm.Yn is different between svmA and svmB")
	} else if !equal2D(a.Xn, b.Xn) {
		t.Errorf("svm.Xn is different between svmA and svmB")
	}
}

func equal(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equal2D(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !equal(a[i], b[i]) {
			return false
		}
	}
	return true
}
