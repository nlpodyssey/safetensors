// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"encoding/json"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

const (
	lastValidDType       = U64
	maxDType       DType = 1<<(unsafe.Sizeof(DType(0))*8) - 1
)

var _ json.Marshaler = DType(0)

var commonTests = []struct {
	DType  DType
	Size   uint64
	String string
	JSON   []byte
}{
	{BOOL, 1, "BOOL", []byte(`"BOOL"`)},
	{U8, 1, "U8", []byte(`"U8"`)},
	{I8, 1, "I8", []byte(`"I8"`)},
	{I16, 2, "I16", []byte(`"I16"`)},
	{U16, 2, "U16", []byte(`"U16"`)},
	{F16, 2, "F16", []byte(`"F16"`)},
	{BF16, 2, "BF16", []byte(`"BF16"`)},
	{I32, 4, "I32", []byte(`"I32"`)},
	{U32, 4, "U32", []byte(`"U32"`)},
	{F32, 4, "F32", []byte(`"F32"`)},
	{F64, 8, "F64", []byte(`"F64"`)},
	{I64, 8, "I64", []byte(`"I64"`)},
	{U64, 8, "U64", []byte(`"U64"`)},
}

func TestDType_Size(t *testing.T) {
	for _, tc := range commonTests {
		assert.Equal(t, tc.Size, tc.DType.Size(), "DType %d (%s)", tc.DType, tc.DType)
	}
	assert.PanicsWithError(t, "cannot get size of invalid DType 200", func() {
		_ = DType(200).Size()
	})

	// Ensure that changes to the enum are noticeable.
	for dt := DType(0); dt <= lastValidDType; dt++ {
		size := dt.Size()
		assert.GreaterOrEqual(t, size, uint64(1))
		assert.LessOrEqual(t, size, uint64(8))
	}
	for dt := maxDType; dt > lastValidDType; dt-- {
		assert.Panicsf(t, func() { _ = dt.Size() }, "DType %d", dt)
	}
}

func TestDType_String(t *testing.T) {
	for _, tc := range commonTests {
		assert.Equal(t, tc.String, tc.DType.String(), "DType %d (%s)", tc.DType, tc.DType)
	}
	assert.PanicsWithError(t, "cannot get string representation of invalid DType 200", func() {
		_ = DType(200).String()
	})

	// Ensure that changes to the enum are noticeable.
	for dt := DType(0); dt <= lastValidDType; dt++ {
		assert.NotEmpty(t, dt.String())
	}
	for dt := maxDType; dt > lastValidDType; dt-- {
		assert.Panicsf(t, func() { _ = dt.String() }, "DType %d", dt)
	}
}

func TestDType_MarshalJSON(t *testing.T) {
	for _, tc := range commonTests {
		got, err := tc.DType.MarshalJSON()
		assert.NoError(t, err)
		assert.Equal(t, tc.JSON, got, "DType %d (%s)", tc.DType, tc.DType)
	}
	{
		_, err := DType(200).MarshalJSON()
		assert.EqualError(t, err, "cannot get JSON string representation of invalid DType 200")
	}

	// Ensure that changes to the enum are noticeable.
	for dt := DType(0); dt <= lastValidDType; dt++ {
		got, err := dt.MarshalJSON()
		assert.NoError(t, err)
		assert.NotEmpty(t, got)
	}
	for dt := maxDType; dt > lastValidDType; dt-- {
		_, err := DType(200).MarshalJSON()
		assert.Error(t, err)
	}
}

func TestParseDType(t *testing.T) {
	for _, tc := range commonTests {
		got, err := ParseDType(tc.String)
		assert.NoErrorf(t, err, "string %q", tc.String)
		assert.Equal(t, tc.DType, got, "string %q", tc.String)
	}
	{
		_, err := ParseDType("foo")
		assert.EqualError(t, err, `invalid DType string value "foo"`)
	}
}
