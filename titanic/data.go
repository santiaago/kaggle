package main

import "strconv"

func passengerFromTrainLine(line []string) passenger {

	survived, err := strconv.ParseBool(line[1])
	if err != nil {
		survived = false
	}
	age, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		age = 25
	}

	sibsp, err := strconv.ParseInt(line[6], 10, 32)
	if err != nil {
		sibsp = 0
	}

	parch, err := strconv.ParseInt(line[7], 10, 32)
	if err != nil {
		parch = 0
	}

	fare, err := strconv.ParseInt(line[9], 10, 32)
	if err != nil {
		fare = 0
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
		int(sibsp),
		int(parch),
		line[8],
		int(fare),
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
	sibsp, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		sibsp = 0
	}
	parch, err := strconv.ParseInt(line[6], 10, 32)
	if err != nil {
		parch = 0
	}
	ticket := line[7]
	fare, err := strconv.ParseInt(line[8], 10, 32)
	if err != nil {
		fare = 0
	}

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
		int(sibsp),
		int(parch),
		ticket,
		int(fare),
		cabin,
		embarked,
	}
	return p
}
