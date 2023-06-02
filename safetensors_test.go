// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeserialize(t *testing.T) {
	serialized := []byte("Y\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"foo":"bar"}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

	loaded, err := Deserialize(serialized)
	require.NoError(t, err)

	assert.Equal(t, 1, loaded.Len())
	assert.Equal(t, []string{"test"}, loaded.Names())

	tensor, ok := loaded.Tensor("test")
	assert.True(t, ok)

	assert.Equal(t, []uint64{2, 2}, tensor.Shape())
	assert.Equal(t, I32, tensor.DType())
	assert.Equal(t, make([]byte, 16), tensor.Data())
}

func TestSerialize(t *testing.T) {
	t.Run("simple serialization", func(t *testing.T) {
		floatData := []float32{0, 1, 2, 3, 4, 5}
		data := make([]byte, 0, len(floatData)*4)
		for _, v := range floatData {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}

		shape := []uint64{1, 2, 3}

		attn0, err := NewTensorView(F32, shape, data)
		require.NoError(t, err)

		metadata := map[string]TensorView{
			"attn.0": attn0,
		}

		out, err := Serialize(metadata, nil)
		require.NoError(t, err)

		expected := []byte{
			64, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 46, 48, 34, 58, 123, 34, 100,
			116, 121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34,
			58, 91, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102, 115,
			101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 0, 0, 0, 0, 0, 0, 128, 63,
			0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64,
		}
		assert.Equal(t, expected, out)

		_, err = Deserialize(out)
		require.NoError(t, err)

		var buf bytes.Buffer
		err = SerializeToWriter(metadata, nil, &buf)
		require.NoError(t, err)
		assert.Equal(t, expected, buf.Bytes())
	})

	t.Run("forced alignment", func(t *testing.T) {
		floatData := []float32{0, 1, 2, 3, 4, 5}
		data := make([]byte, 0, len(floatData)*4)
		for _, v := range floatData {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}

		shape := []uint64{1, 1, 2, 3}

		attn0, err := NewTensorView(F32, shape, data)
		require.NoError(t, err)

		metadata := map[string]TensorView{
			// Smaller string to force misalignment compared to previous test.
			"attn0": attn0,
		}

		out, err := Serialize(metadata, nil)
		require.NoError(t, err)

		expected := []byte{
			72, 0, 0, 0, 0, 0, 0, 0, 123, 34, 97, 116, 116, 110, 48, 34, 58, 123, 34, 100, 116,
			121, 112, 101, 34, 58, 34, 70, 51, 50, 34, 44, 34, 115, 104, 97, 112, 101, 34, 58,
			91, 49, 44, 49, 44, 50, 44, 51, 93, 44, 34, 100, 97, 116, 97, 95, 111, 102, 102,
			// All the 32 are forcing alignement of the tensor data for casting to f32, f64
			// etc..
			115, 101, 116, 115, 34, 58, 91, 48, 44, 50, 52, 93, 125, 125, 32, 32, 32, 32, 32,
			32, 32, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0,
			160, 64,
		}
		assert.Equal(t, expected, out)

		_, err = Deserialize(out)
		require.NoError(t, err)

		var buf bytes.Buffer
		err = SerializeToWriter(metadata, nil, &buf)
		require.NoError(t, err)
		assert.Equal(t, expected, buf.Bytes())
	})
}

func TestGPT2Like(t *testing.T) {
	testCases := []struct {
		name   string
		nHeads int
	}{
		{"gpt2", 12},
		{"gpt2_tiny", 6},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			type tensorDescType struct {
				name  string
				shape []uint64
			}

			tensorsDesc := make([]tensorDescType, 0)
			addTensorDesc := func(name string, shape ...uint64) {
				tensorsDesc = append(tensorsDesc, tensorDescType{name: name, shape: shape})
			}

			addTensorDesc("wte", 50257, 768)
			addTensorDesc("wpe", 1024, 768)
			for i := 0; i < tc.nHeads; i++ {
				pre := fmt.Sprintf("h.%d.", i)
				addTensorDesc(pre+"ln_1.weight", 768)
				addTensorDesc(pre+"ln_1.bias", 768)
				addTensorDesc(pre+"attn.bias", 1, 1, 1024, 1024)
				addTensorDesc(pre+"attn.c_attn.weight", 768, 2304)
				addTensorDesc(pre+"attn.c_attn.bias", 2304)
				addTensorDesc(pre+"attn.c_proj.weight", 768, 768)
				addTensorDesc(pre+"attn.c_proj.bias", 768)
				addTensorDesc(pre+"ln_2.weight", 768)
				addTensorDesc(pre+"ln_2.bias", 768)
				addTensorDesc(pre+"mlp.c_fc.weight", 768, 3072)
				addTensorDesc(pre+"mlp.c_fc.bias", 3072)
				addTensorDesc(pre+"mlp.c_proj.weight", 3072, 768)
				addTensorDesc(pre+"mlp.c_proj.bias", 768)
			}
			addTensorDesc("ln_f.weight", 768)
			addTensorDesc("ln_f.bias", 768)

			dType := F32

			dataSize := uint64(0)
			for _, td := range tensorsDesc {
				dataSize += shapeProd(td.shape)
			}
			dataSize *= dType.Size()

			allData := make([]byte, dataSize)
			metadata := make(map[string]TensorView, len(tensorsDesc))
			offset := uint64(0)
			for _, td := range tensorsDesc {
				n := shapeProd(td.shape)
				buffer := allData[offset : offset+n*dType.Size()]
				tensor, err := NewTensorView(dType, td.shape, buffer)
				require.NoError(t, err)
				metadata[td.name] = tensor
				offset += n
			}

			{
				var buf bytes.Buffer

				out, err := Serialize(metadata, nil)
				require.NoError(t, err)
				_, err = buf.Write(out)
				require.NoError(t, err)

				raw := buf.Bytes()
				_, err = Deserialize(raw)
				require.NoError(t, err)
			}

			// Writer API
			{
				var buf bytes.Buffer

				err := SerializeToWriter(metadata, nil, &buf)
				require.NoError(t, err)

				raw := buf.Bytes()
				_, err = Deserialize(raw)
				require.NoError(t, err)
			}
		})
	}
}

func TestEmptyShapesAllowed(t *testing.T) {
	serialized := []byte("8\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[],"data_offsets":[0,4]}}` +
		"\x00\x00\x00\x00")

	loaded, err := Deserialize(serialized)
	require.NoError(t, err)
	assert.Equal(t, []string{"test"}, loaded.Names())
	tensor, ok := loaded.Tensor("test")
	require.True(t, ok)
	assert.Equal(t, []uint64{}, tensor.shape)
	assert.Equal(t, I32, tensor.DType())
	assert.Equal(t, []byte{0, 0, 0, 0}, tensor.Data())
}

func TestJSONAttack(t *testing.T) {
	tensors := make(map[string]TensorInfo, 10)
	dType := F32
	shape := []uint64{2, 2}
	dataOffsets := [2]uint64{0, 16}

	for i := 0; i < 10; i++ {
		tensors[fmt.Sprintf("weight_%d", i)] = TensorInfo{
			DType:       dType,
			Shape:       shape,
			DataOffsets: dataOffsets,
		}
	}

	serialized, err := json.Marshal(tensors)
	require.NoError(t, err)

	n := uint64(len(serialized))

	var buf bytes.Buffer

	var nbArr [8]byte
	nb := nbArr[:]
	binary.LittleEndian.PutUint64(nb, n)
	_, err = buf.Write(nb)
	require.NoError(t, err)

	_, err = buf.Write(serialized)
	require.NoError(t, err)

	_, err = buf.Write([]byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	require.NoError(t, err)

	_, err = Deserialize(buf.Bytes())
	assert.ErrorContains(t, err, "invalid metadata offset for tensor")
}

func TestMetadataIncompleteBuffer(t *testing.T) {
	t.Run("extra data", func(t *testing.T) {
		serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00extra_bogus_data_for_polyglot_file")

		_, err := Deserialize(serialized)
		assert.EqualError(t, err, "metadata incomplete buffer")
	})

	t.Run("missing data", func(t *testing.T) {
		serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00") // <- missing 2 bytes

		_, err := Deserialize(serialized)
		assert.EqualError(t, err, "metadata incomplete buffer")
	})
}

func TestHeaderTooLarge(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\xff\xff\xff" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}` +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

	_, err := Deserialize(serialized)
	assert.ErrorContains(t, err, "header too large")
}

func TestHeaderTooSmall(t *testing.T) {
	for i := 0; i < 8; i++ {
		data := make([]byte, i)
		_, err := Deserialize(data)
		assert.EqualErrorf(t, err, "header too small", "data len = %d", i)
	}
}

func TestInvalidHeaderLength(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00")
	_, err := Deserialize(serialized)
	assert.EqualError(t, err, "invalid header length")
}

func TestInvalidHeaderNonUTF8(t *testing.T) {
	serialized := []byte("\x01\x00\x00\x00\x00\x00\x00\x00\xff")
	_, err := Deserialize(serialized)
	assert.ErrorContains(t, err, "invalid header deserialization")
}

func TestInvalidHeaderNotJSON(t *testing.T) {
	serialized := []byte("\x01\x00\x00\x00\x00\x00\x00\x00{")
	_, err := Deserialize(serialized)
	assert.ErrorContains(t, err, "invalid header deserialization")
}

func TestZeroSizedTensor(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,0],"data_offsets":[0, 0]}}`)

	loaded, err := Deserialize(serialized)
	require.NoError(t, err)
	require.Equal(t, []string{"test"}, loaded.Names())
	tensor, ok := loaded.Tensor("test")
	require.True(t, ok)
	assert.Equal(t, []uint64{2, 0}, tensor.Shape())
	assert.Equal(t, I32, tensor.DType())
	assert.Equal(t, []byte{}, tensor.Data())
}

func TestInvalidInfo(t *testing.T) {
	serialized := []byte("<\x00\x00\x00\x00\x00\x00\x00" +
		`{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0, 4]}}`)

	_, err := Deserialize(serialized)
	assert.EqualError(t, err, "metadata validation error: info data offsets mismatch")
}

func TestValidationOverflow(t *testing.T) {
	// max uint64 = 18_446_744_073_709_551_615

	t.Run("overflow the shape calculation", func(t *testing.T) {
		serialized := []byte("O\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,18446744073709551614],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

		_, err := Deserialize(serialized)
		assert.ErrorContains(t, err, "metadata validation error: failed to compute num elements from shape: multiplication overflow")
	})

	t.Run("overflow num elements * total shape", func(t *testing.T) {
		serialized := []byte("N\x00\x00\x00\x00\x00\x00\x00" +
			`{"test":{"dtype":"I32","shape":[2,9223372036854775807],"data_offsets":[0,16]}}` +
			"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

		_, err := Deserialize(serialized)
		assert.ErrorContains(t, err, "metadata validation error: failed to compute num bytes from num elements: multiplication overflow")
	})
}

func shapeProd(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 0
	}
	p := shape[0]
	for _, v := range shape[1:] {
		p *= v
	}
	return p
}
