package main

import (
	"fmt"
	"log"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
	"github.com/santiaago/ml/logreg"
	"github.com/santiaago/ml/svm"
)

// testModels runs a model on the data passed in the reader.
// Then makes the predictions and write the predicted data to file using the
// model name.
// testModels run a test file for each model passed in the array.
//
func testModels(models ml.ModelContainers) {
	if !*test {
		return
	}
	w := NewPassengerTestWriter(*testSrc)
	r := NewPassengerReader(*testSrc, NewPassengerTestExtractor())

	if *verbose {
		fmt.Println("Starting testing models")
	}
	var dc data.Container
	var err error

	if dc, err = r.Read(); err != nil {
		log.Println("error when getting data from reader,", err)
	}

	for i, m := range models {
		if *verbose {
			fmt.Printf("\r\ttesting model:%v/%v", i, len(models))
		}
		if m == nil {
			continue
		}
		switch m.Model.(type) {
		case *linreg.LinearRegression:
			if predictions, err := linregTest(m, dc); err == nil {
				w.Write(m.Name, predictions)
			}
		case *logreg.LogisticRegression:
			if predictions, err := logregTest(m, dc); err == nil {
				w.Write(m.Name, predictions)
			}
		case *svm.SVM:
			if predictions, err := svmTest(m, dc); err == nil {
				w.Write(m.Name, predictions)
			}
		}
	}
	if *verbose {
		fmt.Printf("\n")
		fmt.Println("Done testing models")
	}
}
