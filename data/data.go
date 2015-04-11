package data

import "fmt"

type Reader struct {
	Ex Extractor
	Cl Cleaner
}

func (r Reader) Read() ([][]float64, error) {
	d, err := r.Ex.Extract()
	if err != nil {
		return nil, fmt.Errorf("error when extracting data: %v", err)
	}
	return r.Cl.Clean(d)
}

type Extractor interface {
	Extract() (interface{}, error)
}

type Cleaner interface {
	Clean(interface{}) ([][]float64, error)
}
