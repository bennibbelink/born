package cpu

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// Clamp restricts tensor values element-wise to [minBound, maxBound].
func (cpu *CPUBackend) Clamp(input *tensor.RawTensor, minBound, maxBound any) *tensor.RawTensor {
	if minBound == nil || maxBound == nil {
		panic("clamp: min and max bounds cannot be nil")
	}

	if input.DType() == tensor.Float32 {
		checkNaNBounds[float32](minBound, maxBound)
	} else if input.DType() == tensor.Float64 {
		checkNaNBounds[float64](minBound, maxBound)
	}

	var result *tensor.RawTensor
	// when minBound > maxBound set all values to maxBound
	// otherwise, perform normal clamp operation
	switch input.DType() {
	case tensor.Int32:
		if minBound.(int32) > maxBound.(int32) {
			return tensor.Full(input.Shape(), maxBound.(int32), cpu).Raw()
		}
		result = clampGeneric(input, minBound.(int32), maxBound.(int32), cpu)
	case tensor.Int64:
		if minBound.(int64) > maxBound.(int64) {
			return tensor.Full(input.Shape(), maxBound.(int64), cpu).Raw()
		}
		result = clampGeneric(input, minBound.(int64), maxBound.(int64), cpu)
	case tensor.Float32:
		if minBound.(float32) > maxBound.(float32) {
			return tensor.Full(input.Shape(), maxBound.(float32), cpu).Raw()
		}
		result = clampGeneric(input, minBound.(float32), maxBound.(float32), cpu)
	case tensor.Float64:
		if minBound.(float64) > maxBound.(float64) {
			return tensor.Full(input.Shape(), maxBound.(float64), cpu).Raw()
		}
		result = clampGeneric(input, minBound.(float64), maxBound.(float64), cpu)
	default:
		panic("clamp: unsupported dtype (only int32/int64/float32/float64 supported)")
	}

	return result
}

func checkNaNBounds[T float32 | float64](minBound, maxBound any) {
	if math.IsNaN(float64(minBound.(T))) || math.IsNaN(float64(maxBound.(T))) {
		panic("clamp: min and max bounds cannot be NaN")
	}
}

// clampGeneric performs the clamp operation for a specific data type T.
func clampGeneric[T int32 | int64 | float32 | float64](input *tensor.RawTensor, minBound, maxBound T, cpu *CPUBackend) *tensor.RawTensor {
	minValues := tensor.Full(input.Shape(), minBound, cpu).Raw()
	maxValues := tensor.Full(input.Shape(), maxBound, cpu).Raw()

	// Clamp to min: select max(input, minValues)
	minMask := cpu.LowerEqual(input, minValues) // 1 where input <= min, else 0
	clampedMin := cpu.Where(minMask, minValues, input)

	// Clamp to max: select min(clampedMin, maxValues)
	maxMask := cpu.GreaterEqual(clampedMin, maxValues) // 1 where clampedMin >= max, else 0
	clamped := cpu.Where(maxMask, maxValues, clampedMin)

	return clamped
}
