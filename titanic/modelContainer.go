package main

import (
	"fmt"
	"sort"

	"github.com/santiaago/ml/linreg"
)

// todo(santiaago): move to ml
type modelContainer struct {
	Model    *linreg.LinearRegression
	Name     string
	Features []int
}

func NewModelContainer(m *linreg.LinearRegression, n string, features []int) *modelContainer {
	return &modelContainer{m, n, features}
}

type modelContainers []*modelContainer

func (slice modelContainers) Len() int {
	return len(slice)
}

func (slice modelContainers) Less(i, j int) bool {
	return (*slice[i]).Model.Ein() < (*slice[j]).Model.Ein()
}

func (slice modelContainers) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

func (ms modelContainers) PrintTop(n int) {
	sort.Sort(ms)
	for i := 0; i < n && i < len(ms); i++ {
		lr := ms[i].Model
		fmt.Printf("EIn = %f \t%s\n", lr.Ein(), ms[i].Name)
	}
}
