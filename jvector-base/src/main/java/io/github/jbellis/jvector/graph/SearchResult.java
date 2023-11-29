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

package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.util.BitSet;

/**
 * Container class for results of an ANN search, along with associated metrics about the behavior of the search.
 */
public final class SearchResult {
    private final NodeScore[] nodes;
    private final BitSet visited;
    private final int visitedCount;
    private final int exactSimilarities;
    private final int approximateSimilarities;

    public SearchResult(NodeScore[] nodes, BitSet visited, int visitedCount, int exactSimilarities, int approximateSimilarities) {
        this.nodes = nodes;
        this.visited = visited;
        this.visitedCount = visitedCount;
        this.exactSimilarities = exactSimilarities;
        this.approximateSimilarities = approximateSimilarities;
    }

    /**
     * @return the set of graph nodes visited while performing the search.
     * Deliberately not public -- used to "repair" holes left by removing a node.
     */
    BitSet getVisited() {
        return visited;
    }

    /**
     * @return the closest neighbors discovered by the search, sorted best-first
     */
    public NodeScore[] getNodes() {
        return nodes;
    }

    public int getExactSimilarities() {
        return exactSimilarities;
    }

    public int getApproximateSimilarities() {
        return approximateSimilarities;
    }

    /**
     * @return the total number of graph nodes visited while performing the search
     */
    public int getVisitedCount() {
        return visitedCount;
    }

    public static final class NodeScore {
        public final int node;
        public final float score;

        public NodeScore(int node, float score) {
            this.node = node;
            this.score = score;
        }
    }
}
