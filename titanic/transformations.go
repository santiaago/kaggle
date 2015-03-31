package main

import "math"

//    (1, x1, x2, |x1+ x2|)
func nonLinearFeature1(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = math.Abs(x1 + x2)
	return b
}

//    (1, x1, x2, |x1 - x2|)
func nonLinearFeature2(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = math.Abs(x1 - x2)
	return b
}

//    (1, x1, x2, x1 * x2)
func nonLinearFeature3(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	return b
}

//    (1, x1, x2, x1 * x2, , |x1+ x2|)
func nonLinearFeature4(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	b[4] = math.Abs(x1 + x2)
	return b
}

//    (1, x1, x2, x1 * x2, , |x1 - x2|)
func nonLinearFeature5(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	b[4] = math.Abs(x1 - x2)
	return b
}

//    (1, x1, x2, x1^2 + x2^2)
func nonLinearFeature6(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 4)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1*x1 + x2*x2
	return b
}

//    (1, x1, x2, x1 * x2, x1^2 + x2^2)
func nonLinearFeature7(a []float64) []float64 {
	if len(a) != 3 {
		panic(a)
	}
	x1, x2 := a[1], a[2]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x1 * x2
	b[4] = x1*x1 + x2*x2
	return b
}
