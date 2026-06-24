package cpu

import (
	"fmt"
	"math/rand"
	"testing"
)

var dispatchThresholds = []int{4, 8, 16, 32, 64, 128}

// createRandomFloat32Slices returns two 1024-element slices filled with
// random float32 values in [-1, 1), suitable for benchmarking element-wise ops.
func createRandomFloat32Slices(n int) ([]float32, []float32) {
	aSlice := make([]float32, n)
	bSlice := make([]float32, n)
	rng := rand.New(rand.NewSource(0))
	for i := range aSlice {
		aSlice[i] = rng.Float32()*2 - 1
	}
	for i := range bSlice {
		bSlice[i] = rng.Float32()*2 - 1
	}
	return aSlice, bSlice
}

// BenchmarkAddInplaceF32_Scalar benchmarks a[i] += b[i] using the scalar fallback.
func BenchmarkAddInplaceF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)

	saved := simdAddInplaceFloat32
	simdAddInplaceFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		addInplaceFloat32(aSlice, bSlice)
	}
	simdAddInplaceFloat32 = saved
}

// BenchmarkAddInplaceF32_SIMD benchmarks a[i] += b[i] using the SIMD implementation.
// Each sub-benchmark uses a slice length equal to the dispatch threshold so the
// SIMD dispatch condition (len(a) >= threshold) is excercised.
func BenchmarkAddInplaceF32_SIMD(b *testing.B) {
	if simdAddInplaceFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}
	for _, threshold := range dispatchThresholds {
		aSlice, bSlice := createRandomFloat32Slices(threshold)

		b.Run(fmt.Sprintf("Scalar(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdAddInplaceFloat32
			simdAddInplaceFloat32 = nil
			b.ResetTimer()
			for b.Loop() {
				addInplaceFloat32(aCopy, bCopy)
			}
			simdAddInplaceFloat32 = saved
			b.SetBytes(int64(threshold * 4))
		})
		b.Run(fmt.Sprintf("SIMD(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdAddDispatchThreshold
			simdAddDispatchThreshold = threshold
			b.ResetTimer()
			for b.Loop() {
				addInplaceFloat32(aCopy, bCopy)
			}
			simdAddDispatchThreshold = saved
			b.SetBytes(int64(threshold * 4))
		})
	}
}

// BenchmarkSubInplaceF32_Scalar benchmarks a[i] -= b[i] using the scalar fallback.
func BenchmarkSubInplaceF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)

	saved := simdSubInplaceFloat32
	simdSubInplaceFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		subInplaceFloat32(aSlice, bSlice)
	}
	simdSubInplaceFloat32 = saved
}

// BenchmarkSubInplaceF32_SIMD benchmarks a[i] -= b[i] using the SIMD implementation.
// Each sub-benchmark uses a slice length equal to the dispatch threshold so the
// SIMD dispatch condition (len(a) >= threshold) is excercised.
func BenchmarkSubInplaceF32_SIMD(b *testing.B) {
	if simdSubInplaceFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	for _, threshold := range dispatchThresholds {
		aSlice, bSlice := createRandomFloat32Slices(threshold)

		b.Run(fmt.Sprintf("Scalar(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdSubInplaceFloat32
			simdSubInplaceFloat32 = nil
			b.ResetTimer()
			for b.Loop() {
				subInplaceFloat32(aCopy, bCopy)
			}
			simdSubInplaceFloat32 = saved
			b.SetBytes(int64(threshold * 4))
		})
		b.Run(fmt.Sprintf("SIMD(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdSubDispatchThreshold
			simdSubDispatchThreshold = threshold
			b.ResetTimer()
			for b.Loop() {
				subInplaceFloat32(aCopy, bCopy)
			}
			simdSubDispatchThreshold = saved
			b.SetBytes(int64(threshold * 4))
		})
	}
}

// BenchmarkMulInplaceF32_Scalar benchmarks a[i] *= b[i] using the scalar fallback.
func BenchmarkMulInplaceF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)

	saved := simdMulInplaceFloat32
	simdMulInplaceFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		mulInplaceFloat32(aSlice, bSlice)
	}
	simdMulInplaceFloat32 = saved
}

// BenchmarkMulInplaceF32_SIMD benchmarks a[i] *= b[i] using the SIMD implementation.
// Each sub-benchmark uses a slice length equal to the dispatch threshold so the
// SIMD dispatch condition (len(a) >= threshold) is excercised.
func BenchmarkMulInplaceF32_SIMD(b *testing.B) {
	if simdMulInplaceFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	for _, threshold := range dispatchThresholds {
		aSlice, bSlice := createRandomFloat32Slices(threshold)

		b.Run(fmt.Sprintf("Scalar(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdMulInplaceFloat32
			simdMulInplaceFloat32 = nil
			b.ResetTimer()
			for b.Loop() {
				mulInplaceFloat32(aCopy, bCopy)
			}
			simdMulInplaceFloat32 = saved
			b.SetBytes(int64(threshold * 4))
		})
		b.Run(fmt.Sprintf("SIMD(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdMulDispatchThreshold
			simdMulDispatchThreshold = threshold
			b.ResetTimer()
			for b.Loop() {
				mulInplaceFloat32(aCopy, bCopy)
			}
			simdMulDispatchThreshold = saved
			b.SetBytes(int64(threshold * 4))
		})
	}

}

// BenchmarkDivInplaceF32_Scalar benchmarks a[i] /= b[i] using the scalar fallback.
func BenchmarkDivInplaceF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)

	saved := simdDivInplaceFloat32
	simdDivInplaceFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		divInplaceFloat32(aSlice, bSlice)
	}
	simdDivInplaceFloat32 = saved
}

// BenchmarkDivInplaceF32_SIMD benchmarks a[i] /= b[i] using the SIMD implementation.
// Each sub-benchmark uses a slice length equal to the dispatch threshold so the
// SIMD dispatch condition (len(a) >= threshold) is excercised.
func BenchmarkDivInplaceF32_SIMD(b *testing.B) {
	if simdDivInplaceFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	for _, threshold := range dispatchThresholds {
		aSlice, bSlice := createRandomFloat32Slices(threshold)

		b.Run(fmt.Sprintf("Scalar(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdDivInplaceFloat32
			simdDivInplaceFloat32 = nil
			b.ResetTimer()
			for b.Loop() {
				divInplaceFloat32(aCopy, bCopy)
			}
			simdDivInplaceFloat32 = saved
			b.SetBytes(int64(threshold * 4))
		})
		b.Run(fmt.Sprintf("SIMD(len=%d)", threshold), func(b *testing.B) {
			aCopy := make([]float32, len(aSlice))
			bCopy := make([]float32, len(bSlice))
			copy(aCopy, aSlice)
			copy(bCopy, bSlice)

			saved := simdDivDispatchThreshold
			simdDivDispatchThreshold = threshold
			b.ResetTimer()
			for b.Loop() {
				divInplaceFloat32(aCopy, bCopy)
			}
			simdDivDispatchThreshold = saved
			b.SetBytes(int64(threshold * 4))
		})
	}
}

// BenchmarkAddVectorizedF32_Scalar benchmarks dst[i] = a[i] + b[i] using the scalar fallback.
func BenchmarkAddVectorizedF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	saved := simdAddVectorizedFloat32
	simdAddVectorizedFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		addVectorizedFloat32(dst, aSlice, bSlice)
	}
	simdAddVectorizedFloat32 = saved
}

// BenchmarkAddVectorizedF32_SIMD benchmarks dst[i] = a[i] + b[i] using the SIMD implementation.
func BenchmarkAddVectorizedF32_SIMD(b *testing.B) {
	if simdAddVectorizedFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		addVectorizedFloat32(dst, aSlice, bSlice)
	}
}

// BenchmarkSubVectorizedF32_Scalar benchmarks dst[i] = a[i] - b[i] using the scalar fallback.
func BenchmarkSubVectorizedF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	saved := simdSubVectorizedFloat32
	simdSubVectorizedFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		subVectorizedFloat32(dst, aSlice, bSlice)
	}
	simdSubVectorizedFloat32 = saved
}

// BenchmarkSubVectorizedF32_SIMD benchmarks dst[i] = a[i] - b[i] using the SIMD implementation.
func BenchmarkSubVectorizedF32_SIMD(b *testing.B) {
	if simdSubVectorizedFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		subVectorizedFloat32(dst, aSlice, bSlice)
	}
}

// BenchmarkMulVectorizedF32_Scalar benchmarks dst[i] = a[i] * b[i] using the scalar fallback.
func BenchmarkMulVectorizedF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	saved := simdMulVectorizedFloat32
	simdMulVectorizedFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		mulVectorizedFloat32(dst, aSlice, bSlice)
	}
	simdMulVectorizedFloat32 = saved
}

// BenchmarkMulVectorizedF32_SIMD benchmarks dst[i] = a[i] * b[i] using the SIMD implementation.
func BenchmarkMulVectorizedF32_SIMD(b *testing.B) {
	if simdMulVectorizedFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		mulVectorizedFloat32(dst, aSlice, bSlice)
	}
}

// BenchmarkDivVectorizedF32_Scalar benchmarks dst[i] = a[i] / b[i] using the scalar fallback.
func BenchmarkDivVectorizedF32_Scalar(b *testing.B) {
	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	saved := simdDivVectorizedFloat32
	simdDivVectorizedFloat32 = nil
	b.ResetTimer()
	for b.Loop() {
		divVectorizedFloat32(dst, aSlice, bSlice)
	}
	simdDivVectorizedFloat32 = saved
}

// BenchmarkDivVectorizedF32_SIMD benchmarks dst[i] = a[i] / b[i] using the SIMD implementation.
func BenchmarkDivVectorizedF32_SIMD(b *testing.B) {
	if simdDivVectorizedFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	aSlice, bSlice := createRandomFloat32Slices(1024)
	dst := make([]float32, len(aSlice))

	b.ResetTimer()
	for b.Loop() {
		divVectorizedFloat32(dst, aSlice, bSlice)
	}
}
