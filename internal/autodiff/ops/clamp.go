package ops

import (
	"fmt"

	"github.com/born-ml/born/internal/tensor"
)

// ClampOp represents the clamp operation: y = clamp(x, min, max).
//
// Backward pass:
//   - d(clamp(x, min, max))/dx = 1 if min <= x <= max, else 0
//   - grad_input = grad_output * (1 if min <= input <= max, else 0)
type ClampOp struct {
	input    *tensor.RawTensor // x
	minBound any               // min value
	maxBound any               // max value
	output   *tensor.RawTensor // clamp(x, min, max)
}

// NewClampOp creates a new ClampOp.
func NewClampOp(input *tensor.RawTensor, minBound, maxBound any, output *tensor.RawTensor) *ClampOp {
	return &ClampOp{
		input:    input,
		minBound: minBound,
		maxBound: maxBound,
		output:   output,
	}
}

// Backward computes input gradient for clamp.
//
// Since d(clamp(x, min, max))/dx = 1 if min <= x <= max, else 0:
// grad_input = grad_output * (1 if min <= input <= max, else 0).
func (op *ClampOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	input := op.input

	switch input.DType() {
	case tensor.Int32:
		minBound, maxBound := checkBoundsDtype[int32](op.minBound, op.maxBound)
		maskedGrad := clampBackwardGeneric(outputGrad, input, minBound, maxBound, backend)
		return []*tensor.RawTensor{maskedGrad}
	case tensor.Int64:
		minBound, maxBound := checkBoundsDtype[int64](op.minBound, op.maxBound)
		maskedGrad := clampBackwardGeneric(outputGrad, input, minBound, maxBound, backend)
		return []*tensor.RawTensor{maskedGrad}

	case tensor.Float32:
		minBound, maxBound := checkBoundsDtype[float32](op.minBound, op.maxBound)
		maskedGrad := clampBackwardGeneric(outputGrad, input, minBound, maxBound, backend)
		return []*tensor.RawTensor{maskedGrad}

	case tensor.Float64:
		minBound, maxBound := checkBoundsDtype[float64](op.minBound, op.maxBound)
		maskedGrad := clampBackwardGeneric(outputGrad, input, minBound, maxBound, backend)
		return []*tensor.RawTensor{maskedGrad}
	default:
		panic("clamp: unsupported dtype (only int32/int64/float32/float64 supported)")
	}
}

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

func clampBackwardGeneric[T int32 | int64 | float32 | float64](outputGrad, input *tensor.RawTensor, minBound, maxBound T, backend tensor.Backend) *tensor.RawTensor {
	minValues := tensor.Full(input.Shape(), minBound, backend).Raw()
	maxValues := tensor.Full(input.Shape(), maxBound, backend).Raw()

	minMask := backend.GreaterEqual(input, minValues)
	maxMask := backend.LowerEqual(input, maxValues)
	combinedMask := backend.And(minMask, maxMask)

	maskNumeric, err := tensor.Cast(combinedMask, input.DType())
	if err != nil {
		panic(fmt.Sprintf("clamp: failed to cast mask: %v", err))
	}
	maskedGrad := backend.Mul(outputGrad, maskNumeric)

	return maskedGrad
}

// Inputs returns the input tensor [x].
func (op *ClampOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor clamp(x, min, max).
func (op *ClampOp) Output() *tensor.RawTensor {
	return op.output
}
