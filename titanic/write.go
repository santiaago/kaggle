package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/santiaago/ml"
)

func createTempFolder(path string) {

	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err = os.Mkdir(path, 0777); err != nil {
			log.Fatalln(err)
		}
	}
}

func writeEcvRanking(models ml.ModelContainers, name string) {
	if !*ecvRank {
		return
	}

	sort.Sort(ml.ByEcv(models))

	ecv := func(m *ml.ModelContainer) float64 { return m.Model.Ecv() }

	writeRanking(models, name, "model ranking in cross validation error", "Ecv", ecv)
}

func writeEinRanking(models ml.ModelContainers, name string) {
	if !*einRank {
		return
	}
	sort.Sort(ml.ByEin(models))

	ein := func(m *ml.ModelContainer) float64 { return m.Model.Ein() }

	writeRanking(models, name, "model ranking in sample error", "Ein", ein)
}

func writeRanking(models ml.ModelContainers, name, title, errTitle string, modelError func(m *ml.ModelContainer) float64) {

	temp := "data/temp/"
	createTempFolder(temp)

	file, err := os.Create(temp + name)
	defer file.Close()

	if err != nil {
		log.Fatalln(err)
	}

	writer := bufio.NewWriter(file)

	if _, err := writer.WriteString(title + "\n"); err != nil {
		log.Fatalln(err)
	}

	for i, m := range models {
		if m == nil || m.Model == nil {
			continue
		}

		line := fmt.Sprintf("%v\t\t%v = %f\tmodel: %v\n", i, errTitle, modelError(m), m.Name)
		if _, err := writer.WriteString(line); err != nil {
			log.Fatalln(err)
		}
		writer.Flush()
	}
	writer.Flush()
}
