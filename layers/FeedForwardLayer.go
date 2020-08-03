package layers

import (
	"Network/activations"
	"math/rand"
	"sync"
	"time"
)

type FeedForwardLayer struct {
	size          int
	learning_rate float32

	activation_normal func(float32) float32
	activation_slope  func(float32) float32
	softmax           bool

	input_vector []float32
	nodes        []float32
	sloped_sums  []float32
	weights      [][]float32

	input_size  int
	output_size int

	bias_enabled bool
	bias         float32
}

var data_time_global time.Duration

func (layer FeedForwardLayer) Init(params []interface{}) FeedForwardLayer {

	//PARAMETERS - [output_size, input_size, activation, learning_rate, weight_initialization_coefficient, bias]

	//Set layer size
	layer.size = params[1].(int)
	input_size := params[0].(int)
	output_size := params[1].(int)

	layer.input_size = input_size
	layer.output_size = output_size

	//Other parameters
	layer.learning_rate = params[3].(float32)
	layer.bias_enabled = params[5].(bool)

	//Generate nodes
	for i := 0; i < output_size; i++ {
		layer.nodes = append(layer.nodes, float32(0))
		layer.sloped_sums = append(layer.sloped_sums, float32(0))
	}

	//Generate weights
	layer.weights = [][]float32{}
	norm_dist := float64(1) / float64(input_size)

	for i := 0; i < input_size; i++ {
		layer.weights = append(layer.weights, []float32{})

		for n := 0; n < output_size; n++ {
			layer.weights[i] = append(layer.weights[i], float32((rand.Float64()*2-1)*norm_dist)*50)
		}
	}

	//Initialize bias to 0
	if layer.bias_enabled {
		layer.bias = 0
	}

	//Set activation
	switch params[2] {
	case "sigmoid":
		layer.activation_normal = activations.Sigmoid_normal
		layer.activation_slope = activations.Sigmoid_slope
	case "tanh":
		layer.activation_normal = activations.Tanh_normal
		layer.activation_slope = activations.Tanh_slope
	case "linear_relu":
		layer.activation_normal = activations.LinearRelu_normal
		layer.activation_slope = activations.LinearRelu_slope
	case "leaky_relu":
		layer.activation_normal = activations.LeakyRelu_normal
		layer.activation_slope = activations.LeakyRelu_slope
	}

	//Return final layer
	return layer
}

func (layer FeedForwardLayer) ForwardFeed(inputs []float32) (FeedForwardLayer, []float32) {
	//Set input vector for backpropagation
	layer.input_vector = inputs

	//Loop through all node indexes from and to
	var wg sync.WaitGroup

	for node_to := 0; node_to < len(layer.nodes); node_to++ {

		wg.Add(1)

		go func(node_to int) {
			//Calculate initial sum
			sum_value := float32(0)

			for node_from := 0; node_from < len(inputs); node_from++ {
				sum_value += inputs[node_from] * layer.weights[node_from][node_to]
			}

			if layer.bias_enabled {
				sum_value += layer.bias
			}

			//Pass through activation functions
			layer.sloped_sums[node_to] = layer.activation_slope(sum_value)
			layer.nodes[node_to] = layer.activation_normal(sum_value)

			//Indicate that thread is completed
			wg.Done()

		}(node_to)
	}

	wg.Wait()

	return layer, layer.nodes
}

func (layer FeedForwardLayer) BackPropagate(slopes []float32, adjust_weights bool) (FeedForwardLayer, []float32) {
	//Variable init
	slope_cache := float32(0)
	slope_return := []float32{}

	for node_from := 0; node_from < layer.input_size; node_from++ {
		slope_return = append(slope_return, float32(0))
	}

	var wg sync.WaitGroup

	//Loop through all node indexes from and to
	for node_to := 0; node_to < layer.output_size; node_to++ {

		wg.Add(1)

		go func(node_to int) {

			//Calculate node cache and append to slope list
			slope_cache = slopes[node_to] * layer.sloped_sums[node_to]

			for node_from := 0; node_from < layer.input_size; node_from++ {

				//Weight and return slope adjustment
				layer.weights[node_from][node_to] -= slope_cache * layer.input_vector[node_from] * layer.learning_rate
				slope_return[node_from] += slope_cache * layer.weights[node_from][node_to]
			}

			//Bias adjustment
			if layer.bias_enabled {
				layer.bias -= slope_cache * layer.learning_rate
			}

			//Indicate that thread is completed
			wg.Done()

		}(node_to)
	}

	wg.Wait()

	//fmt.Println(layer.bias)

	//Return slopes for input nodes
	return layer, slope_return
}
