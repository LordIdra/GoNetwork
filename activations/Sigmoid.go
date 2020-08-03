package activations

import "math"

func Sigmoid_normal(x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(-x))))
}

func Sigmoid_slope(x float32) float32 {
	y := float32(1 / (1 + math.Exp(float64(-x))))
	return y * (1 - y)
}
