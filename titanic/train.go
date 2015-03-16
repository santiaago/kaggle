package main

import (
	"encoding/csv"
	"log"

	"github.com/santiaago/caltechx.go/linreg"
)

func trainModel(r *csv.Reader) []*linreg.LinearRegression {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}
	var linregs []*linreg.LinearRegression
	linregs = append(linregs, linregSexAge(passengers))
	linregs = append(linregs, linregPClassAge(passengers))
	linregs = append(linregs, linregPClassSex(passengers))
	return linregs

}
