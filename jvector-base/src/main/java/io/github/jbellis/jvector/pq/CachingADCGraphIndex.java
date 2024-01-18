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

package io.github.jbellis.jvector.pq;

import io.github.jbellis.jvector.graph.ApproximateScoreProvider;
import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.graph.NodesIterator;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorByte;
import io.github.jbellis.jvector.vector.types.VectorFloat;

import java.io.IOException;
import java.io.UncheckedIOException;

/**
 * Specialized CachingGraphIndex for OnDiskADCGraphIndex.
 * TODO: Refactor so that caching is pluggable for different GraphIndex implementations.
 */
public class CachingADCGraphIndex implements GraphIndex<VectorFloat<?>>, AutoCloseable, Accountable, ApproximateScoreProvider
{
    private static final int CACHE_DISTANCE = 3;

    private final ADCGraphCache cache;
    private final OnDiskADCGraphIndex<VectorFloat<?>> graph;

    public CachingADCGraphIndex(OnDiskADCGraphIndex<VectorFloat<?>> graph)
    {
        this.graph = graph;
        try {
            this.cache = ADCGraphCache.load(graph, CACHE_DISTANCE);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public int size() {
        return graph.size();
    }

    @Override
    public NodesIterator getNodes() {
        return graph.getNodes();
    }

    @Override
    public View<VectorFloat<?>> getView() {
        return new CachedView(graph.getView());
    }

    @Override
    public int maxDegree() {
        return graph.maxDegree();
    }

    @Override
    public long ramBytesUsed() {
        return graph.ramBytesUsed() + cache.ramBytesUsed();
    }

    @Override
    public void close() throws IOException {
        graph.close();
    }

    @Override
    public NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> query, VectorSimilarityFunction similarityFunction) {
        return graph.approximateScoreFunctionFor(query, similarityFunction);
    }

    public class CachedView implements View<VectorFloat<?>> {
        private final OnDiskADCGraphIndex<VectorFloat<?>>.OnDiskView view;

        public CachedView(OnDiskADCGraphIndex<VectorFloat<?>>.OnDiskView view) {
            this.view = view;
        }

        @Override
        public NodesIterator getNeighborsIterator(int node) {
            var cached = cache.getNode(node);
            if (cached != null) {
                return new NodesIterator.ArrayNodesIterator(cached.neighbors, cached.neighbors.length);
            }
            return view.getNeighborsIterator(node);
        }

        @Override
        public VectorFloat<?> getVector(int node) {
            var cached = cache.getNode(node);
            if (cached != null) {
                return cached.vector;
            }
            return view.getVector(node);
        }

        public VectorByte<?> getPackedNeighbors(int node) {
            var cached = cache.getNode(node);
            if (cached != null) {
                return cached.packedNeighbors;
            }
            return view.getPackedNeighbors(node);
        }

        @Override
        public int size() {
            return view.size();
        }

        @Override
        public int entryNode() {
            return view.entryNode();
        }

        @Override
        public Bits liveNodes() {
            return view.liveNodes();
        }

        @Override
        public void close() throws Exception {
            view.close();
        }
    }
}
