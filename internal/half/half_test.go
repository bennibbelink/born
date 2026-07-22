package half

import (
	"math"
	"testing"
)

func TestFloat16ToFloat32(t *testing.T) {
	cases := []struct {
		name string
		in   uint16
		want float32
	}{
		{"zero", 0x0000, 0},
		{"neg-zero", 0x8000, 0},
		{"one", 0x3C00, 1},
		{"neg-one", 0xBC00, -1},
		{"two", 0x4000, 2},
		{"half", 0x3800, 0.5},
		{"neg-two", 0xC000, -2},
		{"max-normal", 0x7BFF, 65504},
		{"min-positive-normal", 0x0400, float32(math.Ldexp(1, -14))},
		{"smallest-subnormal", 0x0001, float32(math.Ldexp(1, -24))},
	}
	for _, c := range cases {
		if got := Float16ToFloat32(c.in); got != c.want {
			t.Errorf("%s: Float16ToFloat32(0x%04X) = %v, want %v", c.name, c.in, got, c.want)
		}
	}
}

func TestFloat16ToFloat32Special(t *testing.T) {
	if got := Float16ToFloat32(0x7C00); !math.IsInf(float64(got), 1) {
		t.Errorf("+Inf: got %v, want +Inf", got)
	}
	if got := Float16ToFloat32(0xFC00); !math.IsInf(float64(got), -1) {
		t.Errorf("-Inf: got %v, want -Inf", got)
	}
	if got := Float16ToFloat32(0x7E00); !math.IsNaN(float64(got)) {
		t.Errorf("NaN: got %v, want NaN", got)
	}
}
