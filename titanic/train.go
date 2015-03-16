package main

import (
	"encoding/csv"
	"log"

	"github.com/santiaago/caltechx.go/linreg"
)

func trainModels(r *csv.Reader, linregFuncs []func(passengers []passenger) *linreg.LinearRegression) []*linreg.LinearRegression {
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
	for _, f := range linregFuncs {
		linregs = append(linregs, f(passengers))
	}
	return linregs
}

func trainModelsMeta(r *csv.Reader, metaLinregFuncs []func(passengers []passenger) []*linreg.LinearRegression) []*linreg.LinearRegression {
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
	for _, f := range metaLinregFuncs {
		linregs = append(linregs, f(passengers)...)
	}
	return linregs
}
