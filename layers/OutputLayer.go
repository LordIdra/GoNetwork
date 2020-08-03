package layers

type OutputLayer struct {
	size    int
	nodes   []float32
	targets []float32
}

func (layer OutputLayer) Init(params []interface{}) OutputLayer {

	//Set layer size
	layer.size = params[0].(int)

	//Generate layer nodes
	for i := 0; i < params[0].(int); i++ {
		layer.nodes = append(layer.nodes, float32(0))
	}

	//Return final layer
	return layer
}

func (layer OutputLayer) GetCostNormal(targets []float32) []float32 {
	//Calculate costs
	costs := []float32{}
	y := float32(0)
	for i := 0; i < len(layer.nodes); i++ {
		y = (layer.nodes[i] - targets[i])
		costs = append(costs, y*y)
	}

	//Return costs
	return costs
}

func (layer OutputLayer) GetCostSlope(targets []float32, adjust_weights bool) []float32 {
	//Calculate costs
	costs := []float32{}
	for i := 0; i < len(layer.nodes); i++ {
		costs = append(costs, (layer.nodes[i]-targets[i])*2)
	}

	//Return costs
	return costs
}

func (layer OutputLayer) GetOutputs() []float32 {
	//Return final layer outputs
	return layer.nodes
}

func (layer OutputLayer) ForwardFeed(inputs []float32) (OutputLayer, []float32) {
	//Return inputs
	layer.nodes = inputs
	return layer, layer.nodes
}

func (layer OutputLayer) BackPropagate(targets []float32, adjust_weights bool) (OutputLayer, []float32) {
	//Return cost slopes
	return layer, layer.GetCostSlope(targets, adjust_weights)
}
