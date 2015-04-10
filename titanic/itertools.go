package main

// combinations returns an 2D array with all the combinations of the iterable
// array passed as argument with respect to the specified sized 'r'
// From GitHub: https://github.com/ntns
// Written by Nuno Antunes, 2012-08-08
// modified a bit by Santiago Arias
func combinations(iterable []int, r int) (results [][]int) {
	pool := iterable
	n := len(pool)

	if r > n {
		return [][]int{}
	}

	indices := make([]int, r)
	for i := range indices {
		indices[i] = i
	}

	result := make([]int, r)
	for i, el := range indices {
		result[i] = pool[el]
	}

	results = append(results, result)
	for {
		i := r - 1
		for ; i >= 0 && indices[i] == i+n-r; i-- {
		}

		if i < 0 {
			return
		}

		indices[i]++
		for j := i + 1; j < r; j++ {
			indices[j] = indices[j-1] + 1
		}

		for ; i < len(indices); i++ {
			result[i] = pool[indices[i]]
		}
		newRes := make([]int, r)
		copy(newRes, result)
		results = append(results, newRes)
	}
}
