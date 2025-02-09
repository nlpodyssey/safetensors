// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

const maxHeaderSize = 100_000_000

// SafeTensors is a structure owning some metadata to lookup tensors
// on a shared `data` byte-buffer.
type SafeTensors struct {
	metadata Metadata
	data     []byte
}

// Deserialize parses a byte-buffer representing the whole
// safetensor file and returns the deserialized form (no tensor allocation).
func Deserialize(buffer []byte) (SafeTensors, error) {
	n, metadata, err := ReadMetadata(buffer)
	if err != nil {
		return SafeTensors{}, err
	}
	return SafeTensors{
		metadata: metadata,
		data:     buffer[n+8:],
	}, nil
}

// ReadMetadata parses the header and returns the size of the header + parsed
// data, given a byte-buffer representing the whole safetensor file.
func ReadMetadata(buffer []byte) (uint64, Metadata, error) {
	bufferLen := uint64(len(buffer))
	if bufferLen < 8 {
		return 0, Metadata{}, fmt.Errorf("header too small")
	}

	arr := buffer[:8]
	n := binary.LittleEndian.Uint64(arr)
	if n > maxHeaderSize {
		return 0, Metadata{}, fmt.Errorf("header too large: max %d, actual %d", maxHeaderSize, n)
	}

	stop := n + 8
	if stop > bufferLen {
		return 0, Metadata{}, fmt.Errorf("invalid header length")
	}

	var metadata Metadata
	err := json.Unmarshal(buffer[8:stop], &metadata)
	if err != nil {
		return 0, Metadata{}, fmt.Errorf("invalid header deserialization: %w", err)
	}
	bufferEnd, err := metadata.validate()
	if err != nil {
		return 0, Metadata{}, err
	}
	if bufferEnd+8+n != bufferLen {
		return 0, Metadata{}, fmt.Errorf("metadata incomplete buffer")
	}
	return n, metadata, nil
}

// Tensors returns a list of named views of all tensors.
func (st SafeTensors) Tensors() []NamedTensorView {
	tensors := make([]NamedTensorView, len(st.metadata.indexMap))
	for name, index := range st.metadata.indexMap {
		info := &st.metadata.tensors[index]
		tensors[index] = NamedTensorView{
			Name: name,
			TensorView: TensorView{
				dType: info.DType,
				shape: info.Shape,
				data:  st.data[info.DataOffsets[0]:info.DataOffsets[1]],
			},
		}
	}
	return tensors
}

// Tensor allows the user to get the view of a specific tensor by name.
// The returned boolean flag reports whether the tensor was found.
func (st SafeTensors) Tensor(name string) (TensorView, bool) {
	index, ok := st.metadata.indexMap[name]
	if !ok {
		return TensorView{}, false
	}
	info := &st.metadata.tensors[index]
	return TensorView{
		dType: info.DType,
		shape: info.Shape,
		data:  st.data[info.DataOffsets[0]:info.DataOffsets[1]],
	}, true
}

// The Names of all tensors.
func (st SafeTensors) Names() []string {
	names := make([]string, len(st.metadata.indexMap))
	for name, index := range st.metadata.indexMap {
		names[index] = name
	}
	return names
}

// Len returns how many tensors are currently stored within the SafeTensors.
func (st SafeTensors) Len() int {
	return len(st.metadata.tensors)
}

// IsEmpty reports whether the SafeTensors contains any tensor.
func (st SafeTensors) IsEmpty() bool {
	return len(st.metadata.tensors) == 0
}

// Serialize the dictionary of tensors to a byte buffer.
func Serialize[V View](data map[string]V, dataInfo map[string]string) ([]byte, error) {
	pd, tensors, err := prepare(data, dataInfo)
	if err != nil {
		return nil, err
	}
	expectedSize := 8 + pd.n + pd.offset
	buffer := make([]byte, 0, expectedSize)
	buffer = binary.LittleEndian.AppendUint64(buffer, pd.n)
	buffer = append(buffer, pd.headerBytes...)
	for _, tensor := range tensors {
		buffer = append(buffer, tensor.Data()...)
	}
	return buffer, nil
}

// SerializeToWriter the dictionary of tensors to an io.Writer (such as a file).
//
// Compared to Serialize, this procedure reduces the need to allocate the
// whole amount of memory.
func SerializeToWriter[V View](data map[string]V, dataInfo map[string]string, w io.Writer) error {
	pd, tensors, err := prepare(data, dataInfo)
	if err != nil {
		return err
	}

	var nbArr [8]byte
	nb := nbArr[:]
	binary.LittleEndian.PutUint64(nb, pd.n)

	_, err = w.Write(nb)
	if err != nil {
		return err
	}

	_, err = w.Write(pd.headerBytes)
	if err != nil {
		return err
	}

	for _, tensor := range tensors {
		_, err = w.Write(tensor.Data())
		if err != nil {
			return err
		}
	}

	return nil
}

type preparedData struct {
	n           uint64
	headerBytes []byte
	offset      uint64
}

func prepare[V View](dataMap map[string]V, dataInfo map[string]string) (preparedData, []V, error) {
	// Make sure we're sorting by descending dtype alignment,
	// then by name.
	data := make([]NamedView[V], 0, len(dataMap))
	for k, v := range dataMap {
		data = append(data, NamedView[V]{Name: k, View: v})
	}
	sort.Slice(data, func(i, j int) bool {
		l, r := &data[i], &data[j]
		ldt, rdt := l.View.DType(), r.View.DType()
		return ldt > rdt || (ldt == rdt && l.Name < r.Name)
	})

	tensors := make([]V, len(data))
	hMetadata := make([]NamedTensorInfo, len(data))
	offset := uint64(0)

	for i, namedView := range data {
		name, tensor := namedView.Name, namedView.View
		n := tensor.DataLen()
		tensorInfo := TensorInfo{
			DType:       tensor.DType(),
			Shape:       tensor.Shape(),
			DataOffsets: [2]uint64{offset, offset + n},
		}
		offset += n
		hMetadata[i] = NamedTensorInfo{
			Name:       name,
			TensorInfo: tensorInfo,
		}
		tensors[i] = tensor
	}

	metadata := newMetadata(dataInfo, hMetadata)
	metadataBuf, err := json.Marshal(metadata)
	if err != nil {
		return preparedData{}, nil, fmt.Errorf("failed to JSON-marshal metadata: %w", err)
	}

	// Force alignment to 8 bytes.
	extra := (8 - len(metadataBuf)%8) % 8
	if extra > 0 {
		spaces := make([]byte, extra)
		for i := range spaces {
			spaces[i] = ' '
		}
		metadataBuf = append(metadataBuf, spaces...)
	}

	pd := preparedData{
		n:           uint64(len(metadataBuf)),
		headerBytes: metadataBuf,
		offset:      offset,
	}

	return pd, tensors, nil
}
