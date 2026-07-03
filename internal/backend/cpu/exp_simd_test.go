package cpu

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/born-ml/born/internal/tolerance"
)

type expTestCase[T float32 | float64] struct {
	name         string
	srcGenerator func(*rand.Rand) T
}

// TestExpF32_SIMDMatchesScalar verifies that the SIMD float32 Exp matches the scalar result.
func TestExpF32_SIMDMatchesScalar(t *testing.T) {
	if simdExpFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	tol := tolerance.NewDefaultTolerance[float32]()
	rng := rand.New(rand.NewSource(1))

	cases := []expTestCase[float32]{
		{name: "unit", srcGenerator: float32Unit},
		{name: "small", srcGenerator: float32Small},
		{name: "large", srcGenerator: float32Large},
		{name: "special", srcGenerator: floatSpecialCases[float32]},
	}

	for _, c := range cases {
		for _, size := range simdTestSliceLengths {
			t.Run(fmt.Sprintf("%s(size=%d)", c.name, size), func(t *testing.T) {
				src := make([]float32, size)
				dstScalar := make([]float32, size)
				dstSIMD := make([]float32, size)

				for i := range src {
					src[i] = c.srcGenerator(rng)
				}

				expScalar(dstScalar, src)
				simdExpFloat32(dstSIMD, src)

				if err := tolerance.AssertAllApproxEqual(dstScalar, dstSIMD, tol); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

// TestExpF64_SIMDMatchesScalar verifies that the SIMD float64 Exp matches the scalar result.
func TestExpF64_SIMDMatchesScalar(t *testing.T) {
	if simdExpFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	tol := tolerance.NewDefaultTolerance[float64]()
	rng := rand.New(rand.NewSource(1))

	cases := []expTestCase[float64]{
		{name: "unit", srcGenerator: float64Unit},
		{name: "small", srcGenerator: float64Small},
		{name: "large", srcGenerator: float64Large},
		{name: "special", srcGenerator: floatSpecialCases[float64]},
	}

	for _, c := range cases {
		for _, size := range simdTestSliceLengths {
			t.Run(fmt.Sprintf("%s(size=%d)", c.name, size), func(t *testing.T) {
				src := make([]float64, size)
				dstScalar := make([]float64, size)
				dstSIMD := make([]float64, size)

				for i := range src {
					src[i] = c.srcGenerator(rng)
				}

				expScalar(dstScalar, src)
				simdExpFloat64(dstSIMD, src)

				if err := tolerance.AssertAllApproxEqual(dstScalar, dstSIMD, tol); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

// TestExpF32_KnownValues verifies float32 Exp against known expected values including
// zero, negative zero, and the overflow/underflow clamping thresholds.
func TestExpF32_KnownValues(t *testing.T) {
	if simdExpFloat32 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	tol := tolerance.NewDefaultTolerance[float32]()

	// Float32 exp clamping thresholds (from exp_simd_amd64.go).
	const (
		f32ExpHi = 88.37625885009765625
		f32ExpLo = -87.3365478515625
	)

	for _, size := range simdTestSliceLengths {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			src := make([]float32, size)
			dstScalar := make([]float32, size)
			dstSIMD := make([]float32, size)

			// Pattern of edge-case values: zero, -0, boundary thresholds, common values.
			pattern := []float32{
				0,
				float32(math.Copysign(0, -1)), // -0
				float32(f32ExpHi),             // exactly at overflow threshold
				float32(f32ExpHi) - 1,         // just below overflow
				float32(f32ExpLo),             // exactly at underflow threshold
				float32(f32ExpLo) + 1,         // just above underflow
				1,
				-1,
				2,
				-2,
				0.5,
				-0.5,
			}
			for i := range src {
				src[i] = pattern[i%len(pattern)]
			}

			expScalar(dstScalar, src)
			simdExpFloat32(dstSIMD, src)

			if err := tolerance.AssertAllApproxEqual(dstScalar, dstSIMD, tol); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// TestExpF64_KnownValues verifies float64 Exp against known expected values including
// zero, negative zero, and the overflow/underflow clamping thresholds.
func TestExpF64_KnownValues(t *testing.T) {
	if simdExpFloat64 == nil {
		t.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	tol := tolerance.NewDefaultTolerance[float64]()

	// Float64 exp clamping thresholds (from exp_simd_amd64.go).
	const (
		f64ExpHi = 709.782712893384
		f64ExpLo = -708.396418532264
	)

	for _, size := range simdTestSliceLengths {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			src := make([]float64, size)
			dstScalar := make([]float64, size)
			dstSIMD := make([]float64, size)

			// Pattern of edge-case values: zero, -0, boundary thresholds, common values.
			pattern := []float64{
				0,
				math.Copysign(0, -1), // -0
				f64ExpHi,             // exactly at overflow threshold
				f64ExpHi - 1,         // just below overflow
				f64ExpLo,             // exactly at underflow threshold
				f64ExpLo + 1,         // just above underflow
				1,
				-1,
				2,
				-2,
				0.5,
				-0.5,
			}
			for i := range src {
				src[i] = pattern[i%len(pattern)]
			}

			expScalar(dstScalar, src)
			simdExpFloat64(dstSIMD, src)

			if err := tolerance.AssertAllApproxEqual(dstScalar, dstSIMD, tol); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// BenchmarkExpF32_Scalar benchmarks float32 Exp using the scalar fallback.
func BenchmarkExpF32_Scalar(b *testing.B) {
	for _, size := range simdBenchmarkSizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			src := createRandomFloat32Slice(size)
			dst := make([]float32, size)
			b.ResetTimer()
			for b.Loop() {
				expScalar(dst, src)
			}
			b.SetBytes(int64(size * 4))
		})
	}
}

// BenchmarkExpF32_SIMD benchmarks float32 Exp using the SIMD implementation.
func BenchmarkExpF32_SIMD(b *testing.B) {
	if simdExpFloat32 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	for _, size := range simdBenchmarkSizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			src := createRandomFloat32Slice(size)
			dst := make([]float32, size)
			b.ResetTimer()
			for b.Loop() {
				simdExpFloat32(dst, src)
			}
			b.SetBytes(int64(size * 4))
		})
	}
}

// BenchmarkExpF64_Scalar benchmarks float64 Exp using the scalar fallback.
func BenchmarkExpF64_Scalar(b *testing.B) {
	for _, size := range simdBenchmarkSizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			src := createRandomFloat64Slice(size)
			dst := make([]float64, size)
			b.ResetTimer()
			for b.Loop() {
				expScalar(dst, src)
			}
			b.SetBytes(int64(size * 8))
		})
	}
}

// BenchmarkExpF64_SIMD benchmarks float32 Exp using the SIMD implementation.
func BenchmarkExpF64_SIMD(b *testing.B) {
	if simdExpFloat64 == nil {
		b.Skip("SIMD implementation not available (build without GOEXPERIMENT=simd or non-amd64)")
	}

	for _, size := range simdBenchmarkSizes {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			src := createRandomFloat64Slice(size)
			dst := make([]float64, size)
			b.ResetTimer()
			for b.Loop() {
				simdExpFloat64(dst, src)
			}
			b.SetBytes(int64(size * 8))
		})
	}
}
