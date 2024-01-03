/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.github.jbellis.jvector.disk;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.graph.OnHeapGraphIndex;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.pq.ProductQuantization;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.ArrayUtil;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.io.DataOutput;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class OnDiskFusedGraphIndex<T> implements GraphIndex<T>, AutoCloseable, Accountable
{
    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    private final ReaderSupplier readerSupplier;
    private final long neighborsOffset;
    private final int size;
    private final int entryNode;
    private final int maxDegree;
    private final int dimension;
    private final int subspaceCount;

    public OnDiskFusedGraphIndex(ReaderSupplier readerSupplier, long offset)
    {
        this.readerSupplier = readerSupplier;
        this.neighborsOffset = offset + 5 * Integer.BYTES;
        try (var reader = readerSupplier.get()) {
            reader.seek(offset);
            size = reader.readInt();
            dimension = reader.readInt();
            entryNode = reader.readInt();
            maxDegree = reader.readInt();
            subspaceCount = reader.readInt();
        } catch (Exception e) {
            throw new RuntimeException("Error initializing OnDiskGraph at offset " + offset, e);
        }
    }

    /**
     * @return a Map of old to new graph ordinals where the new ordinals are sequential starting at 0,
     * while preserving the original relative ordering in `graph`.  That is, for all node ids i and j,
     * if i &lt; j in `graph` then map[i] &lt; map[j] in the returned map.
     */
    public static <T> Map<Integer, Integer> getSequentialRenumbering(GraphIndex<T> graph) {
        try (var view = graph.getView()) {
            Map<Integer, Integer> oldToNewMap = new HashMap<>();
            int nextOrdinal = 0;
            for (int i = 0; i < view.getIdUpperBound(); i++) {
                if (graph.containsNode(i)) {
                    oldToNewMap.put(i, nextOrdinal++);
                }
            }
            return oldToNewMap;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int maxDegree() {
        return maxDegree;
    }

    /** return a Graph that can be safely queried concurrently */
    public OnDiskFusedGraphIndex<T>.OnDiskView getView()
    {
        return new OnDiskView(readerSupplier.get());
    }

    // TODO: This is fake generic until the reading functionality uses T
    public class OnDiskView implements GraphIndex.View<T>, AutoCloseable
    {
        private final RandomAccessReader reader;
        private final int[] neighbors;
        private final VectorByte<?> packedNeighbors;

        public OnDiskView(RandomAccessReader reader)
        {
            super();
            this.reader = reader;
            this.neighbors = new int[maxDegree];
            this.packedNeighbors = vectorTypeSupport.createByteType(maxDegree * subspaceCount);
        }

        public T getVector(int node) {
            try {
                long offset = neighborsOffset +
                        node * (Integer.BYTES + (long) dimension * Float.BYTES + subspaceCount * maxDegree + (long) Integer.BYTES * (maxDegree + 1)) // earlier entries
                        + Integer.BYTES; // skip the ID
                float[] vector = new float[dimension];
                reader.seek(offset);
                reader.readFully(vector);
                return (T) vector;
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public NodesIterator getNeighborsIterator(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES + subspaceCount * maxDegree) +
                        (node * (long) Integer.BYTES * (maxDegree + 1)));
                int neighborCount = reader.readInt();
                assert neighborCount <= maxDegree : String.format("neighborCount %d > M %d", neighborCount, maxDegree);
                reader.read(neighbors, 0, neighborCount);
                return new NodesIterator.ArrayNodesIterator(neighbors, neighborCount);
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public VectorByte<?> getPackedNeighbors(int node) {
            try {
                reader.seek(neighborsOffset +
                        (node + 1) * (Integer.BYTES + (long) dimension * Float.BYTES)
                        + ((node) * (subspaceCount * maxDegree + Integer.BYTES * (maxDegree + 1))));

                return vectorTypeSupport.readByteType(reader, packedNeighbors.length());
            }
            catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        @Override
        public int size() {
            return OnDiskFusedGraphIndex.this.size();
        }

        @Override
        public int entryNode() {
            return OnDiskFusedGraphIndex.this.entryNode;
        }

        @Override
        public Bits liveNodes() {
            return Bits.ALL;
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }
    }

    @Override
    public NodesIterator getNodes()
    {
        return NodesIterator.fromPrimitiveIterator(IntStream.range(0, size).iterator(), size);
    }

    @Override
    public long ramBytesUsed() {
        return Long.BYTES + 4 * Integer.BYTES;
    }

    public void close() throws IOException {
        readerSupplier.close();
    }

    /**
     * @param graph the graph to write
     * @param vectors the vectors associated with each node
     * @param out the output to write to
     *
     * If any nodes have been deleted, you must use the overload specifying `oldToNewOrdinals` instead.
     */
    public static <T> void write(GraphIndex<T> graph, RandomAccessVectorValues<T> vectors, PQVectors pq, DataOutput out)
            throws IOException
    {
        try (var view = graph.getView()) {
            if (view.getIdUpperBound() > graph.size()) {
                throw new IllegalArgumentException("Graph contains deletes, must specify oldToNewOrdinals map");
            }
        } catch (Exception e) {
            throw new IOException(e);
        }
        write(graph, vectors, getSequentialRenumbering(graph), pq, out);
    }

    /**
     * @param graph the graph to write
     * @param vectors the vectors associated with each node
     * @param oldToNewOrdinals A map from old to new ordinals. If ordinal numbering does not matter,
     *                         you can use `getSequentialRenumbering`, which will "fill in" holes left by
     *                         any deleted nodes.
     * @param out the output to write to
     */
    public static <T> void write(GraphIndex<T> graph,
                                 RandomAccessVectorValues<T> vectors,
                                 Map<Integer, Integer> oldToNewOrdinals,
                                 PQVectors pq,
                                 DataOutput out)
            throws IOException
    {
        if (graph instanceof OnHeapGraphIndex) {
            var ohgi = (OnHeapGraphIndex<T>) graph;
            if (ohgi.getDeletedNodes().cardinality() > 0) {
                throw new IllegalArgumentException("Run builder.cleanup() before writing the graph");
            }
        }
        if (oldToNewOrdinals.size() != graph.size()) {
            throw new IllegalArgumentException(String.format("ordinalMapper size %d does not match graph size %d",
                    oldToNewOrdinals.size(), graph.size()));
        }

        var entriesByNewOrdinal = new ArrayList<>(oldToNewOrdinals.entrySet());
        entriesByNewOrdinal.sort(Comparator.comparingInt(Map.Entry::getValue));
        // the last new ordinal should be size-1
        if (graph.size() > 0 && entriesByNewOrdinal.get(entriesByNewOrdinal.size() - 1).getValue() != graph.size() - 1) {
            throw new IllegalArgumentException("oldToNewOrdinals produced out-of-range entries");
        }

        try (var view = graph.getView()) {
            // graph-level properties
            out.writeInt(graph.size());
            out.writeInt(vectors.dimension());
            out.writeInt(view.entryNode());
            out.writeInt(graph.maxDegree());
            out.writeInt(pq.getCompressedSize());
            byte[] compressedNeighbors = new byte[pq.getCompressedSize() * graph.maxDegree()];

            // for each graph node, write the associated vector and its neighbors
            for (int i = 0; i < oldToNewOrdinals.size(); i++) {
                var entry = entriesByNewOrdinal.get(i);
                int originalOrdinal = entry.getKey();
                int newOrdinal = entry.getValue();
                if (!graph.containsNode(originalOrdinal)) {
                    continue;
                }

                out.writeInt(newOrdinal); // unnecessary, but a reasonable sanity check
                vectorTypeSupport.writeFloatType(out, (VectorFloat<?>) vectors.vectorValue(originalOrdinal));
                var neighbors = view.getNeighborsIterator(originalOrdinal);
                int n = 0;
                var neighborSize = neighbors.size();
                for (; n < neighborSize; n++) {
                    var compressed = pq.get(neighbors.next());
                    // copy into the compressedNeighbors array
                    System.arraycopy(compressed, 0, compressedNeighbors, n * pq.getCompressedSize(), pq.getCompressedSize());
                }
                // fill rest of compressedNeighbors with 0
                Arrays.fill(compressedNeighbors, n * pq.getCompressedSize(), graph.maxDegree() * pq.getCompressedSize(), (byte) 0);

                for (int subspace = 0; subspace < pq.getCompressedSize(); subspace++) {
                    n = 0;
                    for (; n < neighborSize; n++) {
                        out.writeByte(compressedNeighbors[n * pq.getCompressedSize() + subspace]);
                    }
                    for (; n < graph.maxDegree(); n++) {
                        out.writeByte(0);
                    }
                }

                neighbors = view.getNeighborsIterator(originalOrdinal);
                out.writeInt(neighbors.size());
                n = 0;
                for (; n < neighbors.size(); n++) {
                    out.writeInt(oldToNewOrdinals.get(neighbors.nextInt()));
                }
                assert !neighbors.hasNext();

                // pad out to maxEdgesPerNode
                for (; n < graph.maxDegree(); n++) {
                    out.writeInt(-1);
                }
            }
        } catch (Exception e) {
            throw new IOException(e);
        }
    }
}
