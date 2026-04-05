package ops

import (
	"math"

	"github.com/born-ml/born/internal/tensor"
)

// ErfOp represents an element-wise error function operation: output = erf(a).
//
// Backward pass:
//   - d(erf(a))/da = 2/sqrt(pi) * exp(-a²), so grad_a = outputGrad * (2/sqrt(pi) * exp(-a²))
type ErfOp struct {
	input  *tensor.RawTensor // a
	output *tensor.RawTensor // a / b
}

// NewErfOp creates a new ErfOp.
func NewErfOp(a, output *tensor.RawTensor) *ErfOp {
	return &ErfOp{
		input:  a,
		output: output,
	}
}

// Backward computes input gradients for division.
func (op *ErfOp) Backward(outputGrad *tensor.RawTensor, backend tensor.Backend) []*tensor.RawTensor {
	a := op.input

	scale := 2.0 / math.Sqrt(math.Pi)
	// grad_a = outputGrad * (scale * exp(-a²))
	aSquared := backend.Mul(a, a)
	factor := backend.MulScalar(backend.Exp(backend.MulScalar(aSquared, -1)), scale)
	gradA := backend.Mul(outputGrad, factor)
	gradA = reduceBroadcast(gradA, a.Shape(), backend)

	return []*tensor.RawTensor{gradA}
}

// Inputs returns the input tensor [a].
func (op *ErfOp) Inputs() []*tensor.RawTensor {
	return []*tensor.RawTensor{op.input}
}

// Output returns the output tensor a / b.
func (op *ErfOp) Output() *tensor.RawTensor {
	return op.output
}
