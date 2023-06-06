// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dtype

import (
	"encoding"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	_ json.Marshaler           = DType(0)
	_ json.Unmarshaler         = new(DType)
	_ encoding.TextMarshaler   = DType(0)
	_ encoding.TextUnmarshaler = new(DType)
)

var (
	validValues = []struct {
		dType  DType
		size   int
		string string
		json   string
	}{
		{Bool, 1, "BOOL", `"BOOL"`},
		{U8, 1, "U8", `"U8"`},
		{I8, 1, "I8", `"I8"`},
		{U16, 2, "U16", `"U16"`},
		{I16, 2, "I16", `"I16"`},
		{F16, 2, "F16", `"F16"`},
		{BF16, 2, "BF16", `"BF16"`},
		{U32, 4, "U32", `"U32"`},
		{I32, 4, "I32", `"I32"`},
		{F32, 4, "F32", `"F32"`},
		{U64, 8, "U64", `"U64"`},
		{I64, 8, "I64", `"I64"`},
		{F64, 8, "F64", `"F64"`},
	}
	invalidValues = []DType{0, 14, 15, 16, 254, 255}
)

func TestDType_Validate(t *testing.T) {
	for _, tc := range validValues {
		assert.NoError(t, tc.dType.Validate())
	}

	for _, dt := range invalidValues {
		assert.EqualError(t, dt.Validate(), fmt.Sprintf("invalid DType(%d)", dt))
	}
}

func TestDType_String(t *testing.T) {
	for _, tc := range validValues {
		assert.Equal(t, tc.string, tc.dType.String())
	}

	for _, dt := range invalidValues {
		assert.Equal(t, fmt.Sprintf("invalid DType(%d)", dt), dt.String())
	}
}

func TestDType_Size(t *testing.T) {
	for _, tc := range validValues {
		assert.Equal(t, tc.size, tc.dType.Size())
	}

	for _, dt := range invalidValues {
		assert.Equal(t, -1, dt.Size())
	}
}

func TestDType_MarshalJSON(t *testing.T) {
	for _, tc := range validValues {
		b, err := tc.dType.MarshalJSON()
		assert.NoError(t, err)
		assert.Equal(t, []byte(tc.json), b)
	}

	for _, dt := range invalidValues {
		b, err := dt.MarshalJSON()
		assert.EqualError(t, err, fmt.Sprintf("invalid DType(%d)", dt))
		assert.Nil(t, b)
	}
}

func TestDType_UnmarshalJSON(t *testing.T) {
	for _, tc := range validValues {
		var dt DType
		err := dt.UnmarshalJSON([]byte(tc.json))
		assert.NoError(t, err)
		assert.Equal(t, tc.dType, dt)
	}

	var dt DType
	assert.EqualError(t, dt.UnmarshalJSON(nil), `failed to JSON-unmarshal DType from value ""`)
	assert.EqualError(t, dt.UnmarshalJSON([]byte{}), `failed to JSON-unmarshal DType from value ""`)
	assert.EqualError(t, dt.UnmarshalJSON([]byte("foo")), `failed to JSON-unmarshal DType from value "foo"`)
	assert.EqualError(t, dt.UnmarshalJSON([]byte(`"foo"`)), `failed to JSON-unmarshal DType from value "\"foo\""`)
}

func TestDType_MarshalText(t *testing.T) {
	for _, tc := range validValues {
		b, err := tc.dType.MarshalText()
		assert.NoError(t, err)
		assert.Equal(t, []byte(tc.string), b)
	}

	for _, dt := range invalidValues {
		b, err := dt.MarshalText()
		assert.EqualError(t, err, fmt.Sprintf("invalid DType(%d)", dt))
		assert.Nil(t, b)
	}
}

func TestDType_UnmarshalText(t *testing.T) {
	for _, tc := range validValues {
		var dt DType
		err := dt.UnmarshalText([]byte(tc.string))
		assert.NoError(t, err)
		assert.Equal(t, tc.dType, dt)
	}

	var dt DType
	assert.EqualError(t, dt.UnmarshalText(nil), `failed to text-unmarshal DType from value ""`)
	assert.EqualError(t, dt.UnmarshalText([]byte{}), `failed to text-unmarshal DType from value ""`)
	assert.EqualError(t, dt.UnmarshalText([]byte("foo")), `failed to text-unmarshal DType from value "foo"`)
	assert.EqualError(t, dt.UnmarshalText([]byte(`"foo"`)), `failed to text-unmarshal DType from value "\"foo\""`)
}
