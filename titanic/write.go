package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/santiaago/ml"
)

func writeEinRanking(models ml.ModelContainers, name string) {
	if !*einRank {
		return
	}
	temp := "data/temp/"
	if _, err := os.Stat(temp); os.IsNotExist(err) {
		if err = os.Mkdir(temp, 0777); err != nil {
			log.Fatalln(err)
		}
	}

	file, err := os.Create(temp + name)
	defer file.Close()

	if err != nil {
		log.Fatalln(err)
	}

	writer := bufio.NewWriter(file)

	if _, err := writer.WriteString("model ranking in sample error\n"); err != nil {
		log.Fatalln(err)
	}

	sort.Sort(ml.ByEin(models))
	for i, m := range models {
		if m == nil || m.Model == nil {
			continue
		}

		line := fmt.Sprintf("%v\t\tEin = %f\tmodel: %v\n", i, m.Model.Ein(), m.Name)
		if _, err := writer.WriteString(line); err != nil {
			log.Fatalln(err)
		}
		writer.Flush()
	}
	writer.Flush()
}

func writeEcvRanking(models ml.ModelContainers, name string) {
	if !*ecvRank {
		return
	}

	temp := "data/temp/"
	if _, err := os.Stat(temp); os.IsNotExist(err) {
		if err = os.Mkdir(temp, 0777); err != nil {
			log.Fatalln(err)
		}
	}

	file, err := os.Create(temp + name)
	defer file.Close()

	if err != nil {
		log.Fatalln(err)
	}

	writer := bufio.NewWriter(file)

	if _, err := writer.WriteString("model ranking cross validation error\n"); err != nil {
		log.Fatalln(err)
	}

	sort.Sort(ml.ByEcv(models))
	for i, m := range models {
		if m == nil || m.Model == nil {
			continue
		}

		line := fmt.Sprintf("%v\t\tEcv = %f\tmodel: %v\n", i, m.Model.Ecv(), m.Name)
		if _, err := writer.WriteString(line); err != nil {
			log.Fatalln(err)
		}
		writer.Flush()
	}
	writer.Flush()
}
