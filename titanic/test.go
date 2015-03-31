package main

import (
	"encoding/csv"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

func testModels(file string, linregs []*linreg.LinearRegression, mapUsedFeatures map[string][]int) {
	for i := 0; i < len(linregs); i++ {
		if csvfile, err := os.Open(file); err != nil {
			log.Fatalln(err)
		} else {
			reader := csv.NewReader(csvfile)
			testModel(reader, linregs[i], mapUsedFeatures[linregs[i].Name])
		}
	}
}

func testModel(r *csv.Reader, linreg *linreg.LinearRegression, keep []int) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTestLine(rawData[i])
		passengers = append(passengers, p)
	}
	linregTest(linreg, &passengers, keep)
	writeTestModel(passengers, linreg.Name)
}

func writeTestModel(passengers []passenger, name string) {
	if csvfile, err := os.Create("data/temp/" + name); err != nil {
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
