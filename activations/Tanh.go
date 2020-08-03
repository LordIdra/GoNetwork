package activations

import "math"

func Tanh_normal(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func Tanh_slope(x float32) float32 {
	return float32(1 - math.Pow(math.Tanh(float64(x)), 2))
}
