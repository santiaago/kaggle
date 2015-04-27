package main

import "math"

func argmin(args []float64) int {
	min := math.Inf(+1)
	argmin := 0
	for i, arg := range args {
		if arg < min {
			min = arg
			argmin = i
		}
	}
	return argmin
}

func argmax(args []float64) int {
	max := math.Inf(-1)
	argmax := 0
	for i, arg := range args {
		if arg > max {
			max = arg
			argmax = i
		}
	}
	return argmax
}
