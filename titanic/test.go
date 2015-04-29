package main

import (
	"log"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
	"github.com/santiaago/ml/logreg"
)

// testModel runs a linear regression model on the data passed in the reader.
// Then makes the predictions and write the predicted data to file using the
// linear regression model name.
// testModels run a test file for each linear regression model passed in the linreg array.
func testModels(r data.Reader, w data.Writer, models ml.ModelContainers) {
	var dc data.Container
	var err error
	dc, err = r.Read()
	if err != nil {
		log.Println("error when getting data from reader,", err)
	}

	for _, m := range models {
		if m == nil {
			continue
		}
		_, ok := m.Model.(*linreg.LinearRegression)
		if ok {
			predictions, err := linregTest(m, dc)
			if err == nil {
				w.Write(m.Name, predictions)
			}
			continue
		}
		_, ok = m.Model.(*logreg.LogisticRegression)
		if ok {
			predictions, err := logregTest(m, dc)
			if err == nil {
				w.Write(m.Name, predictions)
			}
			continue
		}
	}
}
