package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.pq.PQVectors;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public interface FusedGraphIndex<T> extends GraphIndex<T> {
    public NodeSimilarity.ApproximateScoreFunction approximateFusedScoreFunctionFor(PQVectors pq, T query, VectorSimilarityFunction similarityFunction);
}
