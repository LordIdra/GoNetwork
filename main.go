package main

import (
	"Network/run"

	"bufio"
	"encoding/csv"
	"os"

	"fmt"
	"strconv"
	"time"
)

func max(array []float32) int {
	max_value := float32(-100000000)
	max_index := 0
	for k, v := range array {
		if v > max_value {
			max_value = v
			max_index = k
		}
	}
	return max_index
}

func handwriting() {

	start := time.Now()

	//Load dataset
	datafile, _ := os.Open("datasets/mnist_train.csv")
	reader := csv.NewReader(bufio.NewReader(datafile))
	dataset, _ := reader.ReadAll()

	fmt.Println("DATA LOADED IN ", time.Since(start))
	start = time.Now()

	//Initialize network
	n := run.Network{Layer_names: []string{"InputLayer", "FeedForwardLayer", "FeedForwardLayer", "FeedForwardLayer", "FeedForwardLayer", "OutputLayer"},
		Layer_parameters: [][]interface{}{
			{784},
			{784, 400, "leaky_relu", float32(0.0005), 50, false},
			{400, 200, "leaky_relu", float32(0.0005), 50, false},
			{200, 100, "leaky_relu", float32(0.0005), 50, false},
			{100, 10, "sigmoid", float32(0.0005), 50, true},
			{10}}}
	n = n.Init()

	fmt.Println("NETWORK INITIALIZED IN ", time.Since(start))

	data_time := time.Duration(0)
	forward_time := time.Duration(0)
	backward_time := time.Duration(0)

	//Percentage storage variables
	percentages := []float32{}
	total_percentage := 0

	//Training pass
	fmt.Println("---TRAINING PASS BEGINNING---")
	time.Sleep(500 * time.Millisecond)

	for epoch := 0; epoch < 2; epoch++ {

		for i := 0; i < 60000; i++ {

			start = time.Now()
			//Separate inputs and targets
			data := dataset[i]
			inputs_raw := data[1:785]
			inputs := []float32{}
			app := float64(0)

			//Create input list
			for k := 0; k < len(inputs_raw); k++ {
				app, _ = strconv.ParseFloat(inputs_raw[k], 32)
				inputs = append(inputs, float32(app)/255)
			}

			//Create target list
			target_index_raw, _ := strconv.ParseInt(data[0], 10, 32)
			target_index := int(target_index_raw)

			targets := []float32{}

			for k := 0; k < 10; k++ {
				targets = append(targets, float32(0.01))
			}

			targets[target_index] = 0.99

			data_time += time.Since(start)

			//Forward and backward pass
			start := time.Now()
			n.ForwardFeed(inputs)
			forward_time += time.Since(start)

			start = time.Now()
			n.BackPropagate(targets, true)
			backward_time += time.Since(start)

			//Percentage correct handling
			if max(n.GetOutputs()) == target_index {
				percentages = append(percentages, 1)
			} else {
				percentages = append(percentages, 0)
			}

			if len(percentages) > 500 {
				percentages = percentages[1:501]
			}

			if i%100 == 0 {

				total_percentage = 0

				for _, x := range percentages {
					total_percentage += int(x)
				}
				fmt.Println(max(n.GetOutputs()), target_index)
				fmt.Println(n.GetOutputs())
				fmt.Println(epoch, i, total_percentage/5, "%")
				fmt.Println("")
			}
		}
	}

	fmt.Println("---TRAINING PASS FINISHED---")
	fmt.Println("DATA FORMATTING TIME: ", data_time)
	fmt.Println("FORWARD PASS TIME: ", forward_time)
	fmt.Println("BACKWARD PASS TIME: ", backward_time)

	//TEST PASS

	start = time.Now()

	//Load dataset
	testfile, _ := os.Open("datasets/mnist_test.csv")
	rdr := csv.NewReader(bufio.NewReader(testfile))
	dataset, _ = rdr.ReadAll()

	fmt.Println("DATA LOADED IN ", time.Since(start))

	data_time = time.Duration(0)
	forward_time = time.Duration(0)
	backward_time = time.Duration(0)

	//Percentage storage variables
	percentages = []float32{}
	total_percentage = 0

	//Test pass
	fmt.Println("---TEST PASS BEGINNING---")
	time.Sleep(500 * time.Millisecond)

	for epoch := 0; epoch < 1; epoch++ {
		for i := 1; i < 10000; i++ {

			start = time.Now()
			//Separate inputs and targets
			data := dataset[i]
			inputs_raw := data[1:785]
			inputs := []float32{}
			app := float64(0)

			//Create input list
			for k := 0; k < len(inputs_raw); k++ {
				app, _ = strconv.ParseFloat(inputs_raw[k], 32)
				inputs = append(inputs, float32(app)/255)
			}

			//Create target list
			target_index_raw, _ := strconv.ParseInt(data[0], 10, 32)
			target_index := int(target_index_raw)

			targets := []float32{}

			for k := 0; k < 10; k++ {
				targets = append(targets, float32(0.02))
			}

			targets[target_index] = 0.98

			data_time += time.Since(start)

			//Forward and backward pass
			n.ForwardFeed(inputs)

			//Percentage correct handling
			if max(n.GetOutputs()) == target_index {
				percentages = append(percentages, 1)
			} else {
				percentages = append(percentages, 0)
			}

			if i%1000 == 0 {

				total_percentage = 0

				for _, x := range percentages {
					total_percentage += int(x)
				}
				fmt.Println(max(n.GetOutputs()), target_index)
				fmt.Println(n.GetCosts(targets))
				fmt.Println(epoch, i, float32(total_percentage)/(float32(i)/float32(100)), "%")
				fmt.Println("")
			}
		}
	}
}

func main() {
	handwriting()
}
