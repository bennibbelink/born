package nn

import (
	"github.com/born-ml/born/internal/tensor"
)

// Parameter represents a trainable parameter in a neural network.
//
// Parameters are tensors that require gradient computation during training.
// They typically represent weights and biases of layers.
//
// Example:
//
//	// Create a weight parameter
//	weight := nn.NewParameter("weight", weightTensor)
//
//	// Access the tensor
//	w := weight.Tensor()
//
//	// Get gradient after backward pass
//	grad := weight.Grad()
type Parameter[B tensor.Backend] struct {
	name   string                     // Parameter name (e.g., "weight", "bias")
	tensor *tensor.Tensor[float32, B] // The parameter tensor
	grad   *tensor.Tensor[float32, B] // Gradient tensor (computed during backward pass)
}

// NewParameter creates a new trainable parameter.
//
// The parameter tensor should be initialized before creating the Parameter.
// Gradient will be allocated during the first backward pass.
//
// Parameters:
//   - name: Descriptive name for this parameter (e.g., "linear1.weight")
//   - tensor: The initialized parameter tensor
//
// Returns a new Parameter.
func NewParameter[B tensor.Backend](name string, t *tensor.Tensor[float32, B]) *Parameter[B] {
	// Backend-agnostic persistence (ADR-019 Phase 3): use BackendReleaser when
	// available so parameter.go is decoupled from GPU-specific types.
	if br, ok := any(t.Backend()).(tensor.BackendReleaser); ok {
		br.SetPersistent(t.Raw().BackendData(), true)
	}
	t.Raw().SetGPUPersistent(true) // Legacy path — kept until Phase 4.
	return &Parameter[B]{
		name:   name,
		tensor: t,
		grad:   nil, // Gradient allocated on first backward pass
	}
}

// Name returns the parameter name.
func (p *Parameter[B]) Name() string {
	return p.name
}

// Tensor returns the parameter tensor.
func (p *Parameter[B]) Tensor() *tensor.Tensor[float32, B] {
	return p.tensor
}

// Grad returns the gradient tensor.
//
// Returns nil if no gradient has been computed yet (before backward pass).
func (p *Parameter[B]) Grad() *tensor.Tensor[float32, B] {
	return p.grad
}

// SetGrad sets the gradient tensor.
//
// This is typically called by the optimizer or during backward pass.
func (p *Parameter[B]) SetGrad(grad *tensor.Tensor[float32, B]) {
	p.grad = grad
}

// SetTensor replaces the parameter tensor without reading data to CPU.
//
// The incoming tensor is detached before storage: its grad pointer is cleared
// and requiresGrad is set to false. This prevents the optimizer's computation
// graph (Sub, MulScalar, etc.) from leaking into the next forward pass through
// the parameter's grad field.
//
// The old tensor's GPU buffer is released IMMEDIATELY (GoMLX FinalizeAll pattern)
// instead of waiting for Go's GC which is unaware of GPU memory pressure.
//
// Callers must ensure the new tensor has the same shape and dtype as the
// original, otherwise downstream operations will panic.
func (p *Parameter[B]) SetTensor(t *tensor.Tensor[float32, B]) {
	br, _ := any(t.Backend()).(tensor.BackendReleaser)

	// Release old GPU buffer immediately — do NOT wait for GC.
	if p.tensor != nil {
		// Backend-agnostic release (ADR-019 Phase 3).
		if br != nil {
			br.ReleaseBackendData(p.tensor.Raw().BackendData())
			p.tensor.Raw().SetBackendData(nil)
		}
		p.tensor.Raw().ReleaseGPU() // Legacy path — kept until Phase 4.
	}
	p.tensor = t.Detach()
	// Backend-agnostic persistence (ADR-019 Phase 3).
	if br != nil {
		br.SetPersistent(p.tensor.Raw().BackendData(), true)
	}
	p.tensor.Raw().SetGPUPersistent(true) // Legacy path — kept until Phase 4.
}

// ZeroGrad clears the gradient tensor.
//
// This should be called before each training iteration to avoid
// accumulating gradients from previous iterations.
func (p *Parameter[B]) ZeroGrad() {
	p.grad = nil
}
