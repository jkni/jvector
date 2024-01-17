package io.github.jbellis.jvector.graph;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.types.VectorFloat;

public interface ApproximateScoreProvider {
    /**
     * @return a ScoreFunction suitable for performing approximate search against nodes in the graph.
     * This is often done by searching against compressed or otherwise transformed vectors.
     */
    NodeSimilarity.ApproximateScoreFunction approximateScoreFunctionFor(VectorFloat<?> q, VectorSimilarityFunction similarityFunction);
}
