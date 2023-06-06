// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dtype

import (
	"fmt"
)

// DType represents a safetensors data type.
type DType uint8

const (
	// Bool represents an 8-bit boolean data type.
	Bool DType = iota + 1
	// U8 represents an 8-bit unsigned integer data type.
	U8
	// I8 represents an 8-bit signed integer data type.
	I8
	// U16 represents a 16-bit unsigned integer data type.
	U16
	// I16 represents a 16-bit signed integer data type.
	I16
	// F16 represents a 16-bit half-precision floating point data type.
	F16
	// BF16 represents a 16-bit brain floating point data type.
	BF16
	// U32 represents a 32-bit unsigned integer data type.
	U32
	// I32 represents a 32-bit signed integer data type.
	I32
	// F32 represents a 32-bit floating point data type.
	F32
	// U64 represents a 64-bit unsigned integer data type.
	U64
	// I64 represents a 64-bit signed integer data type.
	I64
	// F64 represents a 64-bit floating point data type.
	F64
)

var (
	dTypeToString = [...]string{
		Bool: "BOOL",
		U8:   "U8",
		I8:   "I8",
		U16:  "U16",
		I16:  "I16",
		F16:  "F16",
		BF16: "BF16",
		U32:  "U32",
		I32:  "I32",
		F32:  "F32",
		U64:  "U64",
		I64:  "I64",
		F64:  "F64",
	}
	dTypeToJSON = [...]string{
		Bool: `"BOOL"`,
		U8:   `"U8"`,
		I8:   `"I8"`,
		U16:  `"U16"`,
		I16:  `"I16"`,
		F16:  `"F16"`,
		BF16: `"BF16"`,
		U32:  `"U32"`,
		I32:  `"I32"`,
		F32:  `"F32"`,
		U64:  `"U64"`,
		I64:  `"I64"`,
		F64:  `"F64"`,
	}
	dTypeToSize = [...]int{
		Bool: 1,
		U8:   1,
		I8:   1,
		U16:  2,
		I16:  2,
		F16:  2,
		BF16: 2,
		U32:  4,
		I32:  4,
		F32:  4,
		U64:  8,
		I64:  8,
		F64:  8,
	}
)

// Validate returns an error if the DType is not valid, otherwise nil.
func (dt DType) Validate() error {
	if dt == 0 || dt > F64 {
		return fmt.Errorf("invalid DType(%d)", dt)
	}
	return nil
}

// String returns a string representation of a DType.
func (dt DType) String() string {
	if err := dt.Validate(); err != nil {
		return err.Error()
	}
	return dTypeToString[dt]
}

// Size returns the size in bytes of one element of this data type,
// or -1 if the DType value is invalid.
func (dt DType) Size() int {
	if err := dt.Validate(); err != nil {
		return -1
	}
	return dTypeToSize[dt]
}

// MarshalJSON satisfies json.Marshaler interface.
func (dt DType) MarshalJSON() ([]byte, error) {
	if err := dt.Validate(); err != nil {
		return nil, err
	}
	return []byte(dTypeToJSON[dt]), nil
}

// UnmarshalJSON satisfies json.Unmarshaler interface.
func (dt *DType) UnmarshalJSON(b []byte) error {
	s := string(b)
	switch s {
	case `"BOOL"`:
		*dt = Bool
	case `"U8"`:
		*dt = U8
	case `"I8"`:
		*dt = I8
	case `"U16"`:
		*dt = U16
	case `"I16"`:
		*dt = I16
	case `"F16"`:
		*dt = F16
	case `"BF16"`:
		*dt = BF16
	case `"U32"`:
		*dt = U32
	case `"I32"`:
		*dt = I32
	case `"F32"`:
		*dt = F32
	case `"U64"`:
		*dt = U64
	case `"I64"`:
		*dt = I64
	case `"F64"`:
		*dt = F64
	default:
		return fmt.Errorf("failed to JSON-unmarshal DType from value %q", s)
	}
	return nil
}

// MarshalText satisfies encoding.TextMarshaler interface.
func (dt DType) MarshalText() ([]byte, error) {
	if err := dt.Validate(); err != nil {
		return nil, err
	}
	return []byte(dTypeToString[dt]), nil
}

// UnmarshalText satisfies encoding.TextUnmarshaler interface.
func (dt *DType) UnmarshalText(text []byte) error {
	s := string(text)
	switch s {
	case "BOOL":
		*dt = Bool
	case "U8":
		*dt = U8
	case "I8":
		*dt = I8
	case "U16":
		*dt = U16
	case "I16":
		*dt = I16
	case "F16":
		*dt = F16
	case "BF16":
		*dt = BF16
	case "U32":
		*dt = U32
	case "I32":
		*dt = I32
	case "F32":
		*dt = F32
	case "U64":
		*dt = U64
	case "I64":
		*dt = I64
	case "F64":
		*dt = F64
	default:
		return fmt.Errorf("failed to text-unmarshal DType from value %q", s)
	}
	return nil
}
