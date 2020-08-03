package activations

func LeakyRelu_normal(x float32) float32 {
	if x > 0 {
		return x
	} else {
		return 0.1 * x
	}
}

func LeakyRelu_slope(x float32) float32 {
	if x > 0 {
		return 1
	} else {
		return 0.1
	}
}
