//go:build windows

package webgpu

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/born-ml/born/internal/tensor"
)

func TestClamp_Float32(t *testing.T) {
	if !IsAvailable() {
		t.Skip("WebGPU not available")
	}

	backend, err := New()
	if err != nil {
		t.Fatalf("failed to create backend: %v", err)
	}
	defer backend.Release()

	tests := []struct {
		name     string
		input    []float32
		shape    tensor.Shape
		min      float32
		max      float32
		expected []float32
	}{
		{
			name:     "basic",
			input:    []float32{-2, -1, 0, 1, 2},
			shape:    tensor.Shape{5},
			min:      -1,
			max:      1,
			expected: []float32{-1, -1, 0, 1, 1},
		},
		{
			name:     "all_below_min",
			input:    []float32{-5, -3, -1},
			shape:    tensor.Shape{3},
			min:      0,
			max:      1,
			expected: []float32{0, 0, 0},
		},
		{
			name:     "all_above_max",
			input:    []float32{2, 3, 5},
			shape:    tensor.Shape{3},
			min:      0,
			max:      1,
			expected: []float32{1, 1, 1},
		},
		{
			name:     "at_boundaries",
			input:    []float32{-1, 0, 0.5, 1, 2},
			shape:    tensor.Shape{5},
			min:      0,
			max:      1,
			expected: []float32{0, 0, 0.5, 1, 1},
		},
		{
			name:     "2d_tensor",
			input:    []float32{-2, -1, 0, 1, 2, 3},
			shape:    tensor.Shape{2, 3},
			min:      0,
			max:      2,
			expected: []float32{0, 0, 0, 1, 2, 2},
		},
		{
			name:     "negative_bounds",
			input:    []float32{-3, -2, -1, 0, 1},
			shape:    tensor.Shape{5},
			min:      -2,
			max:      0,
			expected: []float32{-2, -2, -1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := tensor.NewRaw(tt.shape, tensor.Float32, tensor.WebGPU)
			if err != nil {
				t.Fatalf("failed to create tensor: %v", err)
			}

			byteData := input.Data()
			for i, v := range tt.input {
				bits := math.Float32bits(v)
				byteData[i*4+0] = byte(bits)
				byteData[i*4+1] = byte(bits >> 8)
				byteData[i*4+2] = byte(bits >> 16)
				byteData[i*4+3] = byte(bits >> 24)
			}

			result := backend.Clamp(input, tt.min, tt.max)

			resultData := result.Data()
			actual := make([]float32, len(tt.expected))
			for i := range actual {
				bits := uint32(resultData[i*4+0]) |
					uint32(resultData[i*4+1])<<8 |
					uint32(resultData[i*4+2])<<16 |
					uint32(resultData[i*4+3])<<24
				actual[i] = math.Float32frombits(bits)
			}

			if !compareSlices(t, tt.expected, actual, 1e-5) {
				t.Errorf("Clamp failed: expected %v, got %v", tt.expected, actual)
			}

			if !result.Shape().Equal(tt.shape) {
				t.Errorf("Shape mismatch: expected %v, got %v", tt.shape, result.Shape())
			}
		})
	}
}

func TestClamp_Int32(t *testing.T) {
	if !IsAvailable() {
		t.Skip("WebGPU not available")
	}

	backend, err := New()
	if err != nil {
		t.Fatalf("failed to create backend: %v", err)
	}
	defer backend.Release()

	tests := []struct {
		name     string
		input    []int32
		shape    tensor.Shape
		min      int32
		max      int32
		expected []int32
	}{
		{
			name:     "basic",
			input:    []int32{-2, -1, 0, 1, 2},
			shape:    tensor.Shape{5},
			min:      -1,
			max:      1,
			expected: []int32{-1, -1, 0, 1, 1},
		},
		{
			name:     "all_below_min",
			input:    []int32{-5, -3, -1},
			shape:    tensor.Shape{3},
			min:      0,
			max:      1,
			expected: []int32{0, 0, 0},
		},
		{
			name:     "all_above_max",
			input:    []int32{2, 3, 5},
			shape:    tensor.Shape{3},
			min:      0,
			max:      1,
			expected: []int32{1, 1, 1},
		},
		{
			name:     "2d_tensor",
			input:    []int32{-2, -1, 0, 1, 2, 3},
			shape:    tensor.Shape{2, 3},
			min:      0,
			max:      2,
			expected: []int32{0, 0, 0, 1, 2, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input, err := tensor.NewRaw(tt.shape, tensor.Int32, tensor.WebGPU)
			if err != nil {
				t.Fatalf("failed to create tensor: %v", err)
			}

			byteData := input.Data()
			for i, v := range tt.input {
				binary.LittleEndian.PutUint32(byteData[i*4:i*4+4], uint32(v))
			}

			result := backend.Clamp(input, tt.min, tt.max)

			resultData := result.Data()
			actual := make([]int32, len(tt.expected))
			for i := range actual {
				actual[i] = int32(binary.LittleEndian.Uint32(resultData[i*4 : i*4+4]))
			}

			for i, v := range actual {
				if v != tt.expected[i] {
					t.Errorf("Clamp[%d] = %d, want %d", i, v, tt.expected[i])
				}
			}

			if !result.Shape().Equal(tt.shape) {
				t.Errorf("Shape mismatch: expected %v, got %v", tt.shape, result.Shape())
			}
		})
	}
}

func TestClamp_LargeTensor(t *testing.T) {
	if !IsAvailable() {
		t.Skip("WebGPU not available")
	}

	backend, err := New()
	if err != nil {
		t.Fatalf("failed to create backend: %v", err)
	}
	defer backend.Release()

	size := 1024
	inputData := make([]float32, size)
	expected := make([]float32, size)

	minVal := float32(100)
	maxVal := float32(500)

	for i := 0; i < size; i++ {
		inputData[i] = float32(i - 200) // Range: -200 to 823
		if inputData[i] < minVal {
			expected[i] = minVal
		} else if inputData[i] > maxVal {
			expected[i] = maxVal
		} else {
			expected[i] = inputData[i]
		}
	}

	input, err := tensor.NewRaw(tensor.Shape{size}, tensor.Float32, tensor.WebGPU)
	if err != nil {
		t.Fatalf("failed to create tensor: %v", err)
	}

	byteData := input.Data()
	for i, v := range inputData {
		bits := math.Float32bits(v)
		byteData[i*4+0] = byte(bits)
		byteData[i*4+1] = byte(bits >> 8)
		byteData[i*4+2] = byte(bits >> 16)
		byteData[i*4+3] = byte(bits >> 24)
	}

	result := backend.Clamp(input, minVal, maxVal)

	resultData := result.Data()
	actual := make([]float32, size)
	for i := range actual {
		bits := uint32(resultData[i*4+0]) |
			uint32(resultData[i*4+1])<<8 |
			uint32(resultData[i*4+2])<<16 |
			uint32(resultData[i*4+3])<<24
		actual[i] = math.Float32frombits(bits)
	}

	if !compareSlices(t, expected, actual, 1e-5) {
		t.Errorf("Large Clamp failed")
	}
}
