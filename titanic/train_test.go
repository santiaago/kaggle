package main

import (
	"fmt"
	"testing"

	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/svm"
)

func buildContainer() (data.Container, error) {
	reader := NewPassengerReader("data/test.csv", NewPassengerTrainExtractor())
	return reader.Read()
}
func TestSVMImport(t *testing.T) {

	var dc data.Container
	var err error
	if dc, err = buildContainer(); err != nil {
		t.Error(err)
	}

	features := []int{2, 4, 6, 7, 8, 9, 10}
	fd := dc.FilterWithPredict(features)
	svm := svm.NewSVM()
	svm.K = 20
	svm.Lambda = 0.001
	svm.T = 1000
	svm.InitializeFromData(fd)
	if err := svm.Learn(); err != nil {
		t.Error(err)
	}
	fmt.Println(svm.Wn)
}
