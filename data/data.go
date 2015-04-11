//Package data reads, extracts and cleans data.
package data

// Reader is the interface that wraps the basic read method.
// Read returns a 2 Dimensional array of floats.
// Use it to extract and clean data that you can then pass to a machine learning
// model.
type Reader interface {
	Read() ([][]float64, error)
}
