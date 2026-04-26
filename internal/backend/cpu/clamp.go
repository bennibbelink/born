package cpu

import (
	"fmt"
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// Clamp restricts tensor values element-wise to [minBound, maxBound].
func (cpu *CPUBackend) Clamp(input *tensor.RawTensor, minBound, maxBound any) *tensor.RawTensor {
	if minBound == nil || maxBound == nil {
		panic("clamp: min and max bounds cannot be nil")
	}

	var result *tensor.RawTensor
	// when minBound > maxBound set all values to maxBound
	// otherwise, perform normal clamp operation
	switch input.DType() {
	case tensor.Int32:
		castedMin, castedMax := checkBoundsDtype[int32](minBound, maxBound)
		if castedMin > castedMax {
			return tensor.Full(input.Shape(), castedMax, cpu).Raw()
		}
		result = clampGeneric(input, castedMin, castedMax, cpu)
	case tensor.Int64:
		castedMin, castedMax := checkBoundsDtype[int64](minBound, maxBound)
		if castedMin > castedMax {
			return tensor.Full(input.Shape(), castedMax, cpu).Raw()
		}
		result = clampGeneric(input, castedMin, castedMax, cpu)
	case tensor.Float32:
		castedMin, castedMax := checkBoundsDtype[float32](minBound, maxBound)
		checkNaN(castedMin)
		checkNaN(castedMax)
		if castedMin > castedMax {
			return tensor.Full(input.Shape(), castedMax, cpu).Raw()
		}
		result = clampGeneric(input, castedMin, castedMax, cpu)
	case tensor.Float64:
		castedMin, castedMax := checkBoundsDtype[float64](minBound, maxBound)
		checkNaN(castedMin)
		checkNaN(castedMax)
		if castedMin > castedMax {
			return tensor.Full(input.Shape(), castedMax, cpu).Raw()
		}
		result = clampGeneric(input, castedMin, castedMax, cpu)
	default:
		panic("clamp: unsupported dtype (only int32/int64/float32/float64 supported)")
	}

	return result
}

// checkBoundsDtype checks if the bounds and tensor dtype match.
//
// Panics if they do not match.
func checkBoundsDtype[T int32 | int64 | float32 | float64](minBound, maxBound any) (T, T) {
	minCasted, ok := minBound.(T)
	if !ok {
		panic(fmt.Sprintf("clamp: expected %T min bound, got %T", new(T), minBound))
	}
	maxCasted, ok := maxBound.(T)
	if !ok {
		panic(fmt.Sprintf("clamp: expected %T max bound, got %T", new(T), maxBound))
	}

	return minCasted, maxCasted
}

// checkNaN checks if the value is NaN and panics if it is.
func checkNaN[T float32 | float64](v T) {
	if math.IsNaN(float64(v)) {
		panic("clamp: bounds cannot be NaN")
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
