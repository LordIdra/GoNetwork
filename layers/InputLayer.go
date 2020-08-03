package layers

type InputLayer struct {
	size  int
	nodes []float32
}

func (layer InputLayer) Init(params []interface{}) InputLayer {

	//Set layer size
	layer.size = params[0].(int)

	//Generate layer nodes
	for i := 0; i < params[0].(int); i++ {
		layer.nodes = append(layer.nodes, float32(0))
	}

	//Return final layer
	return layer
}

func (layer InputLayer) ForwardFeed(inputs []float32) (InputLayer, []float32) {
	//Return inputs
	layer.nodes = inputs
	return layer, layer.nodes
}

func (layer InputLayer) BackPropagate(slopes []float32) (InputLayer, []float32) {
	//Placeholder function - doesn't actually do anything, but acts as an endpoint for propagation
	return layer, []float32{}
}
