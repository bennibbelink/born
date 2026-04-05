package cpu

import (
	"fmt"
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// Exp computes element-wise exponential: exp(x).
func (cpu *CPUBackend) Exp(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("exp: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			dst[i] = float32(math.Exp(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			dst[i] = math.Exp(v)
		}
	default:
		panic(fmt.Sprintf("exp: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}

// Log computes element-wise natural logarithm: ln(x).
func (cpu *CPUBackend) Log(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("log: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			if v <= 0 {
				panic(fmt.Sprintf("log: non-positive value at index %d: %f", i, v))
			}
			dst[i] = float32(math.Log(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			if v <= 0 {
				panic(fmt.Sprintf("log: non-positive value at index %d: %f", i, v))
			}
			dst[i] = math.Log(v)
		}
	default:
		panic(fmt.Sprintf("log: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}

// Sqrt computes element-wise square root: sqrt(x).
func (cpu *CPUBackend) Sqrt(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("sqrt: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			if v < 0 {
				panic(fmt.Sprintf("sqrt: negative value at index %d: %f", i, v))
			}
			dst[i] = float32(math.Sqrt(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			if v < 0 {
				panic(fmt.Sprintf("sqrt: negative value at index %d: %f", i, v))
			}
			dst[i] = math.Sqrt(v)
		}
	default:
		panic(fmt.Sprintf("sqrt: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}

// Rsqrt computes element-wise reciprocal square root: 1/sqrt(x).
// This is optimized for use in normalization layers (RMSNorm, LayerNorm).
func (cpu *CPUBackend) Rsqrt(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("rsqrt: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			if v <= 0 {
				panic(fmt.Sprintf("rsqrt: non-positive value at index %d: %f", i, v))
			}
			dst[i] = 1.0 / float32(math.Sqrt(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			if v <= 0 {
				panic(fmt.Sprintf("rsqrt: non-positive value at index %d: %f", i, v))
			}
			dst[i] = 1.0 / math.Sqrt(v)
		}
	default:
		panic(fmt.Sprintf("rsqrt: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}

// Cos computes element-wise cosine: cos(x).
func (cpu *CPUBackend) Cos(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("cos: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			dst[i] = float32(math.Cos(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			dst[i] = math.Cos(v)
		}
	default:
		panic(fmt.Sprintf("cos: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}

// Sin computes element-wise sine: sin(x).
func (cpu *CPUBackend) Sin(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("sin: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			dst[i] = float32(math.Sin(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			dst[i] = math.Sin(v)
		}
	default:
		panic(fmt.Sprintf("sin: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}

// Erf computes element-wise error function: erf(x).
func (cpu *CPUBackend) Erf(x *tensor.RawTensor) *tensor.RawTensor {
	result, err := tensor.NewRaw(x.Shape(), x.DType(), cpu.device)
	if err != nil {
		panic(fmt.Sprintf("erf: %v", err))
	}

	switch x.DType() {
	case tensor.Float32:
		src := x.AsFloat32()
		dst := result.AsFloat32()
		for i, v := range src {
			dst[i] = float32(math.Erf(float64(v)))
		}
	case tensor.Float64:
		src := x.AsFloat64()
		dst := result.AsFloat64()
		for i, v := range src {
			dst[i] = math.Erf(v)
		}
	default:
		panic(fmt.Sprintf("erf: unsupported dtype %s (only float32/float64 supported)", x.DType()))
	}

	return result
}
