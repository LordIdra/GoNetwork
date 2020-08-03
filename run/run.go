package run

import (
	"Network/layers"
)

type Network struct {
	Layer_names      []string
	Layer_parameters [][]interface{}
	Layer_indexes    [][]int

	Input_layer        []layers.InputLayer
	Output_layer       []layers.OutputLayer
	Feed_forward_layer []layers.FeedForwardLayer
}

func (n Network) Init() Network {

	//===PARAMETER LIST===
	//InputLayer - [size]
	//OutputLayer - [size]
	//FeedForwardLayer - [output_size, input_size, activation]

	//Initialize layers and indexes
	for i, x := range n.Layer_names {

		switch x {

		case "InputLayer":

			initialized_layer := layers.InputLayer{}
			initialized_layer = initialized_layer.Init(n.Layer_parameters[i])

			n.Input_layer = append(n.Input_layer, initialized_layer)
			n.Layer_indexes = append(n.Layer_indexes, []int{0, len(n.Input_layer) - 1})

		case "OutputLayer":

			initialized_layer := layers.OutputLayer{}
			initialized_layer = initialized_layer.Init(n.Layer_parameters[i])

			n.Output_layer = append(n.Output_layer, initialized_layer)
			n.Layer_indexes = append(n.Layer_indexes, []int{1, len(n.Output_layer) - 1})

		case "FeedForwardLayer":

			initialized_layer := layers.FeedForwardLayer{}
			initialized_layer = initialized_layer.Init(n.Layer_parameters[i])

			n.Feed_forward_layer = append(n.Feed_forward_layer, initialized_layer)
			n.Layer_indexes = append(n.Layer_indexes, []int{2, len(n.Feed_forward_layer) - 1})
		}
	}

	//Return final network
	return n
}

func (n Network) GetLayerType(index []int) (string, interface{}) {
	//Variable initialization
	var return_value interface{}
	var return_type string

	//Set respective layer types

	switch index[0] {
	case 0:
		return_value = n.Input_layer[index[1]]
		return_type = "InputLayer"

	case 1:
		return_value = n.Output_layer[index[1]]
		return_type = "OutputLayer"

	case 2:
		return_value = n.Feed_forward_layer[index[1]]
		return_type = "FeedForwardLayer"
	}

	//Return final layer list
	return return_type, return_value
}

func (n Network) GetCosts(targets []float32) []float32 {
	//Call get-cost function for output layer. No staircase statement required since we will always be calling the single output layer
	return n.Output_layer[0].GetCostNormal(targets)
}

func (n Network) GetOutputs() []float32 {
	//Return final layer outputs
	return n.Output_layer[0].GetOutputs()
}

func (n Network) ForwardFeed(inputs []float32) Network {
	//Initialize array for storing previous layer result
	new_output := []float32{}

	//Forward feed all layers from 0 to -1
	for _, x := range n.Layer_indexes {

		layer_type, layer_value := n.GetLayerType(x)

		switch layer_type {
		case "InputLayer":
			new_layer := layers.InputLayer{}
			var target_layer = layer_value.(layers.InputLayer)
			new_layer, new_output = target_layer.ForwardFeed(inputs)
			n.Input_layer[x[1]] = new_layer

		case "OutputLayer":
			new_layer := layers.OutputLayer{}
			var target_layer = layer_value.(layers.OutputLayer)
			new_layer, new_output = target_layer.ForwardFeed(new_output)
			n.Output_layer[x[1]] = new_layer

		case "FeedForwardLayer":
			new_layer := layers.FeedForwardLayer{}
			var target_layer = layer_value.(layers.FeedForwardLayer)
			new_layer, new_output = target_layer.ForwardFeed(new_output)
			n.Feed_forward_layer[x[1]] = new_layer
		}
	}

	//Return final network
	return n
}

func (n Network) BackPropagate(targets []float32, adjust_weights bool) Network {
	//Initialize array for storing previous layer result
	new_output := []float32{}

	//Forward feed all layers from 0 to -1
	for i := len(n.Layer_indexes) - 1; i > 0; i-- {

		layer_type, layer_value := n.GetLayerType(n.Layer_indexes[i])

		switch layer_type {
		case "InputLayer":
			new_layer := layers.InputLayer{}
			var target_layer = layer_value.(layers.InputLayer)
			new_layer, new_output = target_layer.BackPropagate(new_output)
			n.Input_layer[n.Layer_indexes[i][1]] = new_layer

		case "OutputLayer":
			new_layer := layers.OutputLayer{}
			var target_layer = layer_value.(layers.OutputLayer)
			new_layer, new_output = target_layer.BackPropagate(targets, adjust_weights)
			n.Output_layer[n.Layer_indexes[i][1]] = new_layer

		case "FeedForwardLayer":
			new_layer := layers.FeedForwardLayer{}
			var target_layer = layer_value.(layers.FeedForwardLayer)
			new_layer, new_output = target_layer.BackPropagate(new_output, adjust_weights)
			n.Feed_forward_layer[n.Layer_indexes[i][1]] = new_layer
		}
	}

	//Return final network
	return n
}
