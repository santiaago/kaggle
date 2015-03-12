package main

import "fmt"

type passenger struct {
	ID       string
	Survived bool
	Pclass   string
	Name     string
	Sex      string
	Age      int
	SibSp    string
	Parch    string
	Ticket   string
	Fare     string
	Cabin    string
	Embarked string
}

func (p passenger) Print() {
	fmt.Println("ID", p.ID)
	fmt.Println("Survived", p.Survived)
	fmt.Println("Pclass", p.Pclass)
	fmt.Println("Name", p.Name)
	fmt.Println("Sex", p.Sex)
	fmt.Println("Age", p.Age)
	fmt.Println("SibSp", p.SibSp)
	fmt.Println("Sex", p.Parch)
	fmt.Println("Ticket", p.Ticket)
	fmt.Println("Cabin", p.Cabin)
	fmt.Println("Fare", p.Fare)
	fmt.Println("Cabin", p.Cabin)
	fmt.Println("Embarked", p.Embarked)
}
