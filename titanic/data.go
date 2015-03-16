package main

import "strconv"

func passengerFromTrainLine(line []string) passenger {

	survived, err := strconv.ParseBool(line[1])
	if err != nil {
		survived = false
	}
	age, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		age = 33
	}

	var embarked string
	if len(line) > 11 {
		embarked = line[11]
	}

	p := passenger{
		line[0],
		survived,
		line[2],
		line[3],
		line[4],
		int(age),
		line[6],
		line[7],
		line[8],
		line[9],
		line[10],
		embarked,
	}
	return p
}

func passengerFromTestLine(line []string) passenger {
	id := line[0]
	pclass := line[1]
	name := line[2]
	sex := line[3]
	age, err := strconv.ParseInt(line[4], 10, 32)
	if err != nil {
		age = 33
	}
	sibsp := line[5]
	parch := line[6]
	ticket := line[7]
	fare := line[8]
	cabin := line[9]
	var embarked string
	if len(line) > 10 {
		embarked = line[10]
	}

	p := passenger{
		id,
		false,
		pclass,
		name,
		sex,
		int(age),
		sibsp,
		parch,
		ticket,
		fare,
		cabin,
		embarked,
	}
	return p
}
