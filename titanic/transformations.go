package main

import "math"

//    (1, x1, x2, |x1+ x2|)
func transform2D1(a []float64) []float64 {
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
func transform2D2(a []float64) []float64 {
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
func transform2D3(a []float64) []float64 {
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
func transform2D4(a []float64) []float64 {
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
func transform2D5(a []float64) []float64 {
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
func transform2D6(a []float64) []float64 {
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
func transform2D7(a []float64) []float64 {
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

//    (1, x1, x2, x3, |x1 + x2 + x3|)
func transform3D1(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = math.Abs(x1 + x2 + x3)
	return b
}

//    (1, x1, x2, x3, |x1 - x2 - x3|)
func transform3D2(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = math.Abs(x1 - x2 - x3)
	return b
}

//    (1, x1, x2, x3, x1 * x2 * x3)
func transform3D3(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	return b
}

//    (1, x1, x2, x3, x1 * x2 * x3, , |x1 + x2 + x3|)
func transform3D4(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	b[5] = math.Abs(x1 + x2 + x3)
	return b
}

//    (1, x1, x2, x3, x1 * x2 * x3, , |x1 - x2 - x3|)
func transform3D5(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	b[5] = math.Abs(x1 - x2 - x3)
	return b
}

//    (1, x1, x2, x3, x1^2 + x2^2 + x3^2)
func transform3D6(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 5)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1*x1 + x2*x2 + x3*x3
	return b
}

//    (1, x1, x2, x3, x1 * x2 * x3, x1^2 + x2^2 + x3^2)
func transform3D7(a []float64) []float64 {
	if len(a) != 4 {
		panic(a)
	}
	x1, x2, x3 := a[1], a[2], a[3]
	b := make([]float64, 6)
	b[0] = float64(1)
	b[1] = x1
	b[2] = x2
	b[3] = x3
	b[4] = x1 * x2 * x3
	b[5] = x1*x1 + x2*x2 + x3*x3
	return b
}
