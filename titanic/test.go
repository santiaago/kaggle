package main

import (
	"encoding/csv"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

// testModels run a test file for each linear regression model passed in the linreg array.
func testModels(file string, linregs []*linreg.LinearRegression, mapUsedFeatures map[string][]int) {
	for i := 0; i < len(linregs); i++ {
		if csvfile, err := os.Open(file); err != nil {
			log.Fatalln(err)
		} else {
			testModel(csv.NewReader(csvfile), linregs[i], mapUsedFeatures[linregs[i].Name])
		}
	}
}

// testModel runs a linear regression model on the data passed in the reader.
// It filters the data with respect to the keep array.
// Then makes the predictions and write the predicted data to file using the
// linear regression model name.
func testModel(r *csv.Reader, linreg *linreg.LinearRegression, keep []int) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTestingRow(rawData[i])
		passengers = append(passengers, p)
	}
	linregTest(linreg, &passengers, keep)
	writeTestModel(passengers, linreg.Name)
}

// writeTestModel writes the passengers data to a file with the
// name passed as param. The data written to file is the PassengerId and
// Survived column.
func writeTestModel(passengers []passenger, name string) {
	temp := "data/temp/"
	if _, err := os.Stat(temp); os.IsNotExist(err) {
		if err = os.Mkdir(temp, 0777); err != nil {
			log.Fatalln(err)
		}
	}

	if csvfile, err := os.Create(temp + name); err != nil {
		log.Fatalln(err)
	} else {
		writer := csv.NewWriter(csvfile)
		// headers
		if err := writer.Write([]string{"PassengerId", "Survived"}); err != nil {
			log.Fatalln(err)
		}
		// data
		for _, passenger := range passengers {
			p := []string{passenger.ID, "0"}
			if passenger.Survived {
				p[1] = "1"
			}
			if err := writer.Write(p); err != nil {
				log.Fatalln(err)
			}
		}
		writer.Flush()
	}
}
