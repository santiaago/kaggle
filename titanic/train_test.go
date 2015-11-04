package main

import (
	"fmt"
	"testing"

	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/svm"
)

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
func TestSVMImport(t *testing.T) {

	var dc data.Container
	var err error
	if dc, err = buildContainer(); err != nil {
		t.Error(err)
	}

	svm := createSVM(t, dc)

	strWn := fmt.Sprintf("From Code Wn: %v", svm.Wn)
	fmt.Println(strWn)
	strEin := fmt.Sprintf("From Code Ein: %v", svm.Ein())
	fmt.Println(strEin)
}
