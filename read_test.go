// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/nlpodyssey/safetensors/float16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var commonDefinitions = map[string]struct {
	dType      dtype.DType
	shape      []int
	typedValue any
	bytes      []byte
}{
	"bool": {
		dtype.Bool, []int{2},
		[]bool{false, true},
		[]byte{0x00, 0x01},
	},
	"u8": {
		dtype.U8, []int{2, 2},
		[]uint8{0, 1, 254, 255},
		[]byte{0x00, 0x01, 0xfe, 0xff},
	},
	"i8": {
		dtype.I8, []int{2, 2},
		[]int8{0, 1, -2, -1},
		[]byte{0x00, 0x01, 0xfe, 0xff},
	},
	"u16": {
		dtype.U16, []int{2, 2},
		[]uint16{0, 1, 65534, 65535},
		[]byte{
			0x00, 0x00 /**/, 0x01, 0x00,
			0xfe, 0xff /**/, 0xff, 0xff,
		},
	},
	"i16": {
		dtype.I16, []int{2, 2},
		[]int16{0, 1, -2, -1},
		[]byte{
			0x00, 0x00 /**/, 0x01, 0x00,
			0xfe, 0xff /**/, 0xff, 0xff,
		},
	},
	"f16": {
		dtype.F16, []int{2, 2},
		[]float16.F16{0x0001, 0x0203, 0x0405, 0x0607},
		[]byte{
			0x01, 0x00 /**/, 0x03, 0x02,
			0x05, 0x04 /**/, 0x07, 0x06,
		},
	},
	"bf16": {
		dtype.BF16, []int{2, 2},
		[]float16.BF16{0x0001, 0x0203, 0x0405, 0x0607},
		[]byte{
			0x01, 0x00 /**/, 0x03, 0x02,
			0x05, 0x04 /**/, 0x07, 0x06,
		},
	},
	"u32": {
		dtype.U32, []int{2, 2},
		[]uint32{1, 2, 4294967294, 4294967295},
		[]byte{
			0x01, 0x00, 0x00, 0x00 /**/, 0x02, 0x00, 0x00, 0x00,
			0xfe, 0xff, 0xff, 0xff /**/, 0xff, 0xff, 0xff, 0xff,
		},
	},
	"i32": {
		dtype.I32, []int{2, 2},
		[]int32{1, 2, -2, -1},
		[]byte{
			0x01, 0x00, 0x00, 0x00 /**/, 0x02, 0x00, 0x00, 0x00,
			0xfe, 0xff, 0xff, 0xff /**/, 0xff, 0xff, 0xff, 0xff,
		},
	},
	"f32": {
		dtype.F32, []int{2, 2},
		[]float32{1, 2, -1, -2},
		[]byte{
			0x00, 0x00, 0x80, 0x3f /**/, 0x00, 0x00, 0x00, 0x40,
			0x00, 0x00, 0x80, 0xbf /**/, 0x00, 0x00, 0x00, 0xc0,
		},
	},
	"u64": {
		dtype.U64, []int{2, 1},
		[]uint64{1, 18446744073709551615},
		[]byte{
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		},
	},
	"i64": {
		dtype.I64, []int{1, 2},
		[]int64{1, -1},
		[]byte{
			0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		},
	},
	"f64": {
		dtype.F64, []int{2},
		[]float64{1, -1},
		[]byte{
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xbf,
		},
	},
	"zero data": {
		dtype.U8, []int{0},
		[]uint8{},
		nil,
	},
	"no shape scalar": {
		dtype.U8, nil,
		[]uint8{42},
		[]byte{42},
	},
}

func makeCommonData(t *testing.T) []byte {
	header := make(map[string]any, len(commonDefinitions))
	byteBuffer := bytes.NewBuffer(nil)

	for name, def := range commonDefinitions {
		shape := def.shape
		if shape == nil {
			shape = []int{}
		}
		header[name] = map[string]any{
			"dtype":        def.dType.String(),
			"shape":        shape,
			"data_offsets": [2]int{byteBuffer.Len(), byteBuffer.Len() + len(def.bytes)},
		}
		_, err := byteBuffer.Write(def.bytes)
		require.NoError(t, err)
	}

	header["__metadata__"] = map[string]string{"meta...": "data!"}

	jsonHeader, err := json.Marshal(header)
	require.NoError(t, err)

	return makeData(string(jsonHeader), byteBuffer.Bytes())
}

func makeData(jsonHeader string, byteBuffer []byte) []byte {
	data := make([]byte, 8+len(jsonHeader)+len(byteBuffer))
	binary.LittleEndian.PutUint64(data, uint64(len(jsonHeader)))
	copy(data[8:8+len(jsonHeader)], jsonHeader)
	copy(data[8+len(jsonHeader):], byteBuffer)
	return data
}

func TestReadAll(t *testing.T) {
	sfData := makeCommonData(t)

	st, err := ReadAll(bytes.NewReader(sfData), 0)
	require.NoError(t, err)

	assert.Equal(t, map[string]string{"meta...": "data!"}, st.Metadata)

	tensors := st.Tensors
	assert.Len(t, tensors, len(commonDefinitions))

	for name, def := range commonDefinitions {
		t.Run(fmt.Sprintf("tensor %q", name), func(t *testing.T) {
			var tensor *Tensor
			for i := range tensors {
				if tensors[i].Name() == name {
					tensor = &tensors[i]
				}
			}
			if tensor == nil {
				t.Fatal("tensor not found")
			}
			assert.Equal(t, def.dType, tensor.DType())
			if len(def.shape) == 0 {
				assert.Nil(t, tensor.Shape())
			} else {
				assert.Equal(t, def.shape, tensor.Shape())
			}
			assert.Equal(t, def.typedValue, tensor.Data())
		})
	}
}

func TestReadAllRaw(t *testing.T) {
	sfData := makeCommonData(t)

	st, err := ReadAllRaw(bytes.NewReader(sfData), 0)
	require.NoError(t, err)

	assert.Equal(t, map[string]string{"meta...": "data!"}, st.Metadata)

	tensors := st.Tensors
	assert.Len(t, tensors, len(commonDefinitions))

	for name, def := range commonDefinitions {
		t.Run(fmt.Sprintf("tensor %q", name), func(t *testing.T) {
			var tensor *RawTensor
			for i := range tensors {
				if tensors[i].Name() == name {
					tensor = &tensors[i]
				}
			}
			if tensor == nil {
				t.Fatal("tensor not found")
			}
			assert.Equal(t, def.dType, tensor.DType())
			if len(def.shape) == 0 {
				assert.Equal(t, []int{}, tensor.Shape())
			} else {
				assert.Equal(t, def.shape, tensor.Shape())
			}
			assert.Equal(t, def.bytes, tensor.Data())
		})
	}
}
