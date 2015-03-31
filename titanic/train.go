package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/santiaago/caltechx.go/linreg"
)

// trainModels returns:
// * an array of trained LinearRegression models.
// It uses the file passed as param as training data.
// It trains multiple models using different techniques:
// * trainSpecificModels
// * trainModelsByFeatrueCombination
// We return an array of all the linear regression models trained.
func trainModels(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	linregs, usedFeaturesSpecific := trainSpecificModels(file)
	//linregsByComb, usedFeaturesByComb := trainModelsByFeatureCombination(file)
	//linregsWithTransform, usedFeaturesWithTransform := trainModelsWithTransform(file)

	usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesSpecific...)
	//usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesByComb...)
	//usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeaturesWithTransform...)

	//linregs = append(linregs, linregsByComb...)
	//linregs = append(linregs, linregsWithTransform...)
	return
}

// trainSpecificModels trains the following models:
// * linregSexAge
// * linregPClassAge
// * linregPClassSex
// * linregSexAgePClass
// It returns an array of all the linear regression models trained.
func trainSpecificModels(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	funcs := []func(passengers []passenger) (*linreg.LinearRegression, []int){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		linregs, usedFeaturesPerModel = trainModelsByFuncs(csv.NewReader(csvfile), funcs)
	}
	return
}

// trainModelsByFeatureCombination returns:
// * an array of linearRegression models
// It makes a model for every combinations of features present in the data.
// Each feature corresponds to a column in the data set.
func trainModelsByFeatureCombination(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {

	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		funcs := linregVectorsOfInterval()
		linregsOf := trainModelsByMetaFuncs(csv.NewReader(csvfile), funcs)

		linregs = append(linregs, linregsOf...)
	}
	return
}

// trainModelsWithTransform returns:
// * an array of linearRegression models
// It makes a model with the following transformations:
// * todo
func trainModelsWithTransform(file string) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {
	//func linregSexAgePClass(passengers []passenger) *linreg.LinearRegression {
	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		r := csv.NewReader(csvfile)

		var rawData [][]string
		var err error
		if rawData, err = r.ReadAll(); err != nil {
			log.Fatalln(err)
			return nil, nil
		}
		passengers := []passenger{}
		for i := 1; i < len(rawData); i++ {
			p := passengerFromTrainLine(rawData[i])
			passengers = append(passengers, p)
		}

		var data [][]float64
		for i := 0; i < len(passengers); i++ {
			p := passengers[i]
			var survived float64
			if p.Survived {
				survived = float64(1)
			}

			var sex float64
			if p.Sex == "female" {
				sex = float64(1)
			}
			d := []float64{sex, float64(p.Age), survived}
			data = append(data, d)
		}

		funcs := []func([]float64) []float64{
			nonLinearFeature1,
			nonLinearFeature2,
			nonLinearFeature3,
			nonLinearFeature4,
			nonLinearFeature5,
			nonLinearFeature6,
			nonLinearFeature7,
		}
		index := 0
		for _, f := range funcs {
			linreg := linreg.NewLinearRegression()
			linreg.Name = fmt.Sprintf("Sex Age transformed %d", index)
			linreg.InitializeFromData(data)
			linreg.TransformFunction = f
			linreg.ApplyTransformation()
			linreg.Learn()
			fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
			linregs = append(linregs, linreg)
			index++
		}
	}
	return
}

// trainModelsByFuncs returns an array of linear regression models with respect
// to an array of functions passed as arguments.
// Those function takes as argument the passengers data and return a linear
// regression model.
func trainModelsByFuncs(r *csv.Reader, funcs []func(passengers []passenger) (*linreg.LinearRegression, []int)) (linregs []*linreg.LinearRegression, usedFeaturesPerModel [][]int) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil, nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}

	for _, f := range funcs {
		linreg, usedFeatures := f(passengers)
		linregs = append(linregs, linreg)
		usedFeaturesPerModel = append(usedFeaturesPerModel, usedFeatures)
	}
	return
}

// trainModelsByMetaFuncs returns an array of linear regression models with
// respect to an array of linear regression functions passed as arguments.
// Those functions takes as argument the passengers data and
// return an array of linear regression model.
func trainModelsByMetaFuncs(r *csv.Reader, metaLinregFuncs []func(passengers []passenger) []*linreg.LinearRegression) []*linreg.LinearRegression {
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
