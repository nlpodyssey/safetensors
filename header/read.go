// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"

	"github.com/nlpodyssey/safetensors/dtype"
)

type rawDecodedHeader map[string]map[string]any

const metadataKey = "__metadata__"

// Read reads and parses from "r" the initial part of a safetensors
// data stream.
//
// Note that after successfully reading and parsing, NO validation is
// performed on the obtained Header.
//
// This function will fail to read header data larger than math.MaxInt.
// The caller is responsible for guarding against reading data up to a lower
// limit, for example for protection against bad/corrupted data or specific
// attacks. This can be done by providing a reader implementation with a
// limiting mechanism in place. For example, see io.LimitedReader.
func Read(r io.Reader) (Header, error) {
	size, err := readHeaderSize(r)
	switch {
	case err != nil:
		return Header{}, err
	case size < 2: // a bare minimum header is "{}"
		return Header{}, fmt.Errorf("header size too small: %d", size)
	case size > math.MaxInt-8: // 8 bytes are the uint64 "size", already read
		return Header{}, fmt.Errorf("header size too large: %d", size)
	}

	raw, err := readAndDecodeJSON(r, int64(size))
	if err != nil {
		return Header{}, fmt.Errorf("failed to JSON-decode header: %w", err)
	}

	h, err := convertRawHeader(raw)
	if err != nil {
		return Header{}, err
	}

	h.ByteBufferOffset = 8 + int(size) // take into account "size" uint64 bytes
	return h, nil
}

func readHeaderSize(r io.Reader) (uint64, error) {
	var arr [8]byte
	b := arr[:]
	if _, err := io.ReadFull(r, b); err != nil {
		return 0, fmt.Errorf("failed to read header size: %w", err)
	}
	return binary.LittleEndian.Uint64(b), nil
}

func readAndDecodeJSON(r io.Reader, size int64) (rawDecodedHeader, error) {
	dec := json.NewDecoder(&io.LimitedReader{R: r, N: size})
	dec.UseNumber()

	var raw rawDecodedHeader
	if err := dec.Decode(&raw); err != nil {
		return nil, err
	}
	// take care of possible padding spaces after JSON object
	if off := dec.InputOffset(); off != size {
		if _, err := dec.Token(); err == nil {
			return nil, fmt.Errorf("unexpected data at byte offset %d", off)
		} else if err != io.EOF {
			return nil, err
		}
	}
	return raw, nil
}

func convertRawHeader(raw rawDecodedHeader) (h Header, err error) {
	if rawMeta, ok := raw[metadataKey]; ok {
		delete(raw, metadataKey)
		if h.Metadata, err = convertRawMetadata(rawMeta); err != nil {
			return
		}
	}
	h.Tensors, err = convertRawTensors(raw)
	return
}

func convertRawMetadata(raw map[string]any) (Metadata, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	metadata := make(Metadata, len(raw))
	for key, rawVal := range raw {
		var ok bool
		if metadata[key], ok = rawVal.(string); !ok {
			return nil, fmt.Errorf("failed to interpret header metadata: found non-string value for key %q", key)
		}
	}
	return metadata, nil
}

func convertRawTensors(raw rawDecodedHeader) (TensorMap, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	tensors := make(TensorMap, len(raw))
	for key, rawVal := range raw {
		var err error
		if tensors[key], err = convertRawTensor(key, rawVal); err != nil {
			return nil, fmt.Errorf("failed to interpret header tensor %q: %w", key, err)
		}
	}
	return tensors, nil
}

func convertRawTensor(name string, raw map[string]any) (t Tensor, err error) {
	t.Name = name
	if t.DType, err = convertRawTensorDType(raw); err != nil {
		return
	}
	if t.Shape, err = convertRawTensorShape(raw); err != nil {
		return
	}
	if t.DataOffsets, err = convertRawDataOffsets(raw); err != nil {
		return
	}
	if len(raw) != 3 {
		err = errors.New("JSON object contains unknown keys")
	}
	return
}

func convertRawTensorDType(raw map[string]any) (dtype.DType, error) {
	rawDType, ok := raw["dtype"]
	if !ok {
		return 0, errors.New(`"dtype" is missing`)
	}
	strDType, ok := rawDType.(string)
	if !ok {
		return 0, errors.New(`found non-string "dtype" value`)
	}
	var dt dtype.DType
	if err := dt.UnmarshalText([]byte(strDType)); err != nil {
		return 0, fmt.Errorf(`invalid "dtype" value: %q`, strDType)
	}
	return dt, nil
}

func convertRawTensorShape(raw map[string]any) ([]int, error) {
	rawShape, ok := raw["shape"]
	if !ok {
		return nil, fmt.Errorf(`"shape" is missing`)
	}
	rawSlice, ok := rawShape.([]any)
	if !ok {
		return nil, errors.New(`found non-array "shape" value`)
	}
	shape := make([]int, len(rawSlice))
	for i, rawItem := range rawSlice {
		var err error
		if shape[i], err = convertNonNegInt(rawItem); err != nil {
			return nil, fmt.Errorf(`failed to interpret "shape" value at index %d: %w`, i, err)
		}
	}
	return shape, nil
}

func convertRawDataOffsets(raw map[string]any) (DataOffsets, error) {
	rawDataOffsets, ok := raw["data_offsets"]
	if !ok {
		return DataOffsets{}, fmt.Errorf(`"data_offsets" is missing`)
	}
	rawSlice, ok := rawDataOffsets.([]any)
	if !ok {
		return DataOffsets{}, errors.New(`found non-array "data_offsets" value`)
	}
	if l := len(rawSlice); l != 2 {
		return DataOffsets{}, fmt.Errorf(`bad "data_offsets" length: expected 2, actual %d`, l)
	}
	var parsed [2]int
	for i, rawItem := range rawSlice {
		var err error
		if parsed[i], err = convertNonNegInt(rawItem); err != nil {
			return DataOffsets{}, fmt.Errorf(`failed to interpret "data_offsets" value at index %d: %w`, i, err)
		}
	}
	return DataOffsets{Begin: parsed[0], End: parsed[1]}, nil
}

func convertNonNegInt(value any) (int, error) {
	jNum, ok := value.(json.Number)
	if !ok {
		return 0, errors.New("value is not a number")
	}
	num, err := strconv.ParseInt(jNum.String(), 10, strconv.IntSize)
	if err != nil {
		return 0, fmt.Errorf("failed to convert value %q to int: %w", jNum.String(), err)
	}
	if num < 0 {
		return 0, fmt.Errorf("value is negative: %d", num)
	}
	return int(num), nil
}
