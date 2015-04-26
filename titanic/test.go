package main

import (
	"log"

	"github.com/santiaago/kaggle/data"
)

// testModel runs a linear regression model on the data passed in the reader.
// It filters the data with respect to the keep array.
// Then makes the predictions and write the predicted data to file using the
// linear regression model name.
// testModels run a test file for each linear regression model passed in the linreg array.
func testModels(r data.Reader, w data.Writer, models []*modelContainer) {
	var dc data.Container
	var err error
	dc, err = r.Read()
	if err != nil {
		log.Println("error when getting data from reader,", err)
	}

	for _, m := range models {
		predictions := linregTest(m, dc)
		w.Write(m.Name, predictions)
	}
}
