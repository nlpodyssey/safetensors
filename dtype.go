// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import "fmt"

// DType identifies a data type.
type DType uint8

// DType values MUST be in increasing alignment order.
const (
	// BOOL represents a boolean type.
	BOOL DType = iota
	// U8 represents an unsigned byte type.
	U8
	// I8 represents a signed byte type.
	I8
	// I16 represents a 16-bit signed integer type.
	I16
	// U16 represents a 16-bit unsigned integer type.
	U16
	// F16 represents a half-precision (16-bit) floating point type.
	F16
	// BF16 represents a brain (16-bit) floating point type.
	BF16
	// I32 represents a 32-bit signed integer type.
	I32
	// U32 represents a 32-bit unsigned integer type.
	U32
	// F32 represents a 32-bit floating point type.
	F32
	// F64 represents a 64-bit floating point type.
	F64
	// I64 represents a 64-bit signed integer type.
	I64
	// U64 represents a 64-bit unsigned integer type.
	U64
)

var (
	dTypeToSize = [...]uint64{
		BOOL: 1,
		U8:   1,
		I8:   1,
		I16:  2,
		U16:  2,
		F16:  2,
		BF16: 2,
		I32:  4,
		U32:  4,
		F32:  4,
		F64:  8,
		I64:  8,
		U64:  8,
	}
	dTypeToString = [...]string{
		BOOL: "BOOL",
		U8:   "U8",
		I8:   "I8",
		I16:  "I16",
		U16:  "U16",
		F16:  "F16",
		BF16: "BF16",
		I32:  "I32",
		U32:  "U32",
		F32:  "F32",
		F64:  "F64",
		I64:  "I64",
		U64:  "U64",
	}
	dTypeToJSON = [...][]byte{
		BOOL: []byte(`"BOOL"`),
		U8:   []byte(`"U8"`),
		I8:   []byte(`"I8"`),
		I16:  []byte(`"I16"`),
		U16:  []byte(`"U16"`),
		F16:  []byte(`"F16"`),
		BF16: []byte(`"BF16"`),
		I32:  []byte(`"I32"`),
		U32:  []byte(`"U32"`),
		F32:  []byte(`"F32"`),
		F64:  []byte(`"F64"`),
		I64:  []byte(`"I64"`),
		U64:  []byte(`"U64"`),
	}
	stringToDType = map[string]DType{
		"BOOL": BOOL,
		"U8":   U8,
		"I8":   I8,
		"I16":  I16,
		"U16":  U16,
		"F16":  F16,
		"BF16": BF16,
		"I32":  I32,
		"U32":  U32,
		"F32":  F32,
		"F64":  F64,
		"I64":  I64,
		"U64":  U64,
	}
)

// Size returns the size in bytes of one element of this data type.
// It panics if the DType value is invalid.
func (dt DType) Size() uint64 {
	if dt >= DType(len(dTypeToSize)) {
		panic(fmt.Errorf("cannot get size of invalid DType %d", dt))
	}
	return dTypeToSize[dt]
}

// String representation of a DType.
func (dt DType) String() string {
	if dt >= DType(len(dTypeToString)) {
		panic(fmt.Errorf("cannot get string representation of invalid DType %d", dt))
	}
	return dTypeToString[dt]
}

func (dt DType) MarshalJSON() ([]byte, error) {
	if dt >= DType(len(dTypeToJSON)) {
		return nil, fmt.Errorf("cannot get JSON string representation of invalid DType %d", dt)
	}
	return dTypeToJSON[dt], nil
}

// ParseDType attempts to parse a DType value from string.
func ParseDType(s string) (DType, error) {
	dt, ok := stringToDType[s]
	if !ok {
		return 0, fmt.Errorf("invalid DType string value %q", s)
	}
	return dt, nil
}
