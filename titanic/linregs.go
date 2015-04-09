package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linreg"
)

type regressions []*linreg.LinearRegression

func (slice regressions) Len() int {
	return len(slice)
}

func (slice regressions) Less(i, j int) bool {
	return (*slice[i]).Ein() < (*slice[j]).Ein()
}

func (slice regressions) Swap(i, j int) {
	slice[i], slice[j] = slice[j], slice[i]
}

func (slice regressions) Print(top int) {
	for i := 0; i < top; i++ {
		lr := slice[i]
		fmt.Printf("EIn = %f \t%s\n", lr.Ein(), lr.Name)
	}
}
