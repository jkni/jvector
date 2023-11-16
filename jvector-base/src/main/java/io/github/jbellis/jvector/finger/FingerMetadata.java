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

package io.github.jbellis.jvector.finger;

import io.github.jbellis.jvector.graph.GraphIndex;
import io.github.jbellis.jvector.graph.NodeSimilarity;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.util.BitSet;
import io.github.jbellis.jvector.util.PoolingSupport;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class FingerMetadata {

    public final float[][] cBasisProjections;
    public final float[] cSquaredNorms;
    public final float[][] dProjScalarFactor;
    public final float[][] dResSquared;
    public final float[][] dRes;
    public final long[][] sgnDResTB;
    public final LshBasis basis;
    public final GraphIndex<float[]> index;
    public final RandomAccessVectorValues<float[]> ravv;
    public final float[] cachedCosine;

    // low-rank param is fake, always 64
    public static FingerMetadata compute(GraphIndex<float[]> index, RandomAccessVectorValues<float[]> ravv, int lowRank) {
        var ravvPool = ravv.isValueShared() ? PoolingSupport.newThreadBased(ravv::copy) : PoolingSupport.newNoPooling(ravv);
        var ravvCopy = ravvPool.get().get();
        var size = ravvCopy.size();
        var cSquaredNorms = computeCSquaredNorms(ravvPool.get().get());
        var indexView = index.getView();
        var dProjScalarFactor = new float[size][];
        var dResSquared = new float[size][];
        var dResiduals = new float[size][];
        var sgnDResTB = new long[size][];
        var cBasisProjections = new float[size][];
        var basis = LshBasis.computeFromResiduals(ravvCopy, ravvCopy.vectorValue(0).length, lowRank);
        for (int i = 0; i < size; i++) {
            float[] c = ravvCopy.vectorValue(i);
            cBasisProjections[i]= basis.project(c);
            var neighbors = indexView.getNeighborsIterator(i);
            var neighborCount = neighbors.size();
            dProjScalarFactor[i] = new float[neighborCount];
            dResSquared[i] = new float[neighborCount];
            dResiduals[i] = new float[neighborCount];
            sgnDResTB[i] = new long[neighborCount * 2];
            // for neighbor in neighbors:

            for (int n = 0; n < neighborCount; n++) {
                var neighbor = neighbors.next();
                float[] d = ravvCopy.vectorValue(neighbor);
                float projScale = VectorUtil.dotProduct(d, c) / VectorUtil.dotProduct(c, c);
                float[] dProj = new float[d.length];
                float[] dRes = new float[d.length];
                for (int dI = 0; dI < d.length; dI++) {
                    dProj[dI] = projScale * c[dI];
                    dRes[dI] = d[dI] - dProj[dI];
                }
                dProjScalarFactor[i][n] = projScale;
                var dResSquaredNorm = VectorUtil.dotProduct(dRes, dRes);
                dResSquared[i][n] = dResSquaredNorm;
                dResiduals[i][n] = (float) Math.sqrt(dResSquaredNorm);
                var dResB = basis.project(dRes);
                // TODO: this is hardcoded for rank 64
                long encoded = 0;
                for (int k = 0; k < 64; k++) {
                    if (dResB[k] >= 0) {
                        encoded |= 1L << k;
                    }
                }
                sgnDResTB[i][n * 2] = encoded;
                encoded = 0;
                for (int k = 64; k < 128; k++) {
                    if (dResB[k] >= 0) {
                        encoded |= 1L << k - 64;
                    }
                }
                sgnDResTB[i][n * 2 + 1] = encoded;
            }
        }

        return new FingerMetadata(cSquaredNorms, cBasisProjections, dProjScalarFactor, dResSquared, dResiduals, sgnDResTB, basis, index, ravv);
    }

    private static float[] computeCSquaredNorms(RandomAccessVectorValues<float[]> ravv) {
        float[] cSquaredNorms = new float[ravv.size()];
        for (int targetOrd = 0; targetOrd < ravv.size(); targetOrd++) {
            float[] v = ravv.vectorValue(targetOrd);
            cSquaredNorms[targetOrd] = VectorUtil.dotProduct(v, v);
        }
        return cSquaredNorms;
    }

    static class LshBasis {
        public final float[][] basis;

        public LshBasis(float[][] basis) {
            this.basis = basis;
        }

        public float[] project(float[] v) {
            int dim = basis.length;
            float[] projection = new float[dim];
            for (int i = 0; i < dim; i++) {
                projection[i] = VectorUtil.dotProduct(basis[i], v); // JKNI TODO: optimize
            }
            return projection;
        }

        private static float[][] toFloatMatrix(RealMatrix m) {
            float[][] result = new float[m.getRowDimension()][];
            for (int i = 0; i < m.getRowDimension(); i++) {
                result[i] = toFloatArray(m.getRowVector(i));
            }
            return result;
        }

        static void computeOuterProduct(float[] a, float[] b, float[][] result) {
            int dimension = a.length;
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    result[i][j] = a[i] * b[j];
                }
            }
        }

        static void addInPlace(float[][] m1, float[][] m2) {
            int rows = m1.length;
            int cols = m1[0].length;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    m1[i][j] += m2[i][j];
                }
            }
        }

        static void multiplyInPlace(float[][] m, float f) {
            int rows = m.length;
            int cols = m[0].length;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    m[i][j] *= f;
                }
            }
        }

        static float[][] incrementalCovariance(List<float[]> data, int dimension) {
            // TODO: JKNI: is this real?

            // we don't center our data -- it's already centered-ish by construction, and
            // doing an extra transform is more work that we don't want to do for each query

            // Initialize sum of squares matrix and outerProduct scratch space
            float[][] sumOfSquares = new float[dimension][dimension];
            float[][] outerProduct = new float[dimension][dimension];

            // Iterate over data
            int count = 0;
            for (float[] vector : data) {
                computeOuterProduct(vector, vector, outerProduct);
                addInPlace(sumOfSquares, outerProduct);
                count++;
            }

            // Compute raw covariance matrix
            multiplyInPlace(sumOfSquares, 1.0f / count);
            return sumOfSquares;
        }

        static <T> LshBasis computeFromResiduals(RandomAccessVectorValues<float[]> ravv, int dataDimensions, int lshDimensions) {
            var ravvCopy = ravv.isValueShared() ? PoolingSupport.newThreadBased(ravv::copy) : PoolingSupport.newNoPooling(ravv);
            var vectors = IntStream.range(0, ravv.size()).parallel()
                    .mapToObj(targetOrd -> {
                        try (var pooledRavv = ravvCopy.get()) {
                            var localRavv = pooledRavv.get();
                            float[] v = localRavv.vectorValue(targetOrd);
                            return localRavv.isValueShared() ? Arrays.copyOf(v, v.length) : v;
                        }
                    })
                    .collect(Collectors.toList());
            float[][] covarianceMatrix = incrementalCovariance(vectors, dataDimensions);

            // Convert float[][] to double[][]
            double[][] covarianceMatrixDouble = new double[covarianceMatrix.length][covarianceMatrix[0].length];
            for (int i = 0; i < covarianceMatrix.length; i++) {
                for (int j = 0; j < covarianceMatrix[i].length; j++) {
                    covarianceMatrixDouble[i][j] = covarianceMatrix[i][j];
                }
            }

            // Compute the eigen decomposition of the covariance matrix.
            RealMatrix covarianceMatrixRM = MatrixUtils.createRealMatrix(covarianceMatrixDouble);
            EigenDecomposition eig = new EigenDecomposition(covarianceMatrixRM);

            // The LSH basis is given by the eigenvectors corresponding to the r largest eigenvalues.
            // Get the eigenvalues and the matrix of eigenvectors.
            double[] eigenvalues = eig.getRealEigenvalues();
            RealMatrix eigenvectors = eig.getV();

            // Create a stream of indices [0, 1, 2, ..., n-1], sort them by corresponding eigenvalue in descending order,
            // and select the top lshDimensions indices.
            int[] topIndices = IntStream.range(0, eigenvalues.length)
                    .boxed()
                    .sorted(Comparator.comparingDouble(i -> -eigenvalues[i]))
                    .mapToInt(Integer::intValue)
                    .limit(lshDimensions)
                    .toArray();

            // Extract the corresponding eigenvectors.
            float[][] basis = new float[lshDimensions][];
            for (int i = 0; i < lshDimensions; i++) {
                RealVector eigenvector = eigenvectors.getColumnVector(topIndices[i]);
                basis[i] = toFloatArray(eigenvector);
            }

            return new LshBasis(basis);
        }

        private static float[] toFloatArray(RealVector v) {
            float[] result = new float[v.getDimension()];
            for (int i = 0; i < result.length; i++) {
                result[i] = (float) v.getEntry(i);
            }
            return result;
        }
    }

    private FingerMetadata(float[] cSquaredNorms, float[][] cBasisProjections, float[][] dProjScalarFactor, float[][] dResSquared,
                           float[][] dRes, long[][] sgnDResTB, LshBasis basis, GraphIndex<float[]> index, RandomAccessVectorValues<float[]> ravv) {
        this.cSquaredNorms = cSquaredNorms;
        this.cBasisProjections = cBasisProjections;
        this.dProjScalarFactor = dProjScalarFactor;
        this.dResSquared = dResSquared;
        this.dRes = dRes;
        this.sgnDResTB = sgnDResTB;
        this.basis = basis;
        this.index = index;
        this.ravv = ravv.copy();
        this.cachedCosine = new float[129];
        for (int i = 0; i < 129; i++) {
            cachedCosine[i] = (float) Math.cos(Math.PI * i / 129);
        }
    }

    public NodeSimilarity.EstimatedNeighborsScoreFunction estimatedScoreFunctionFor(float[] q, VectorSimilarityFunction similarityFunction) {
        // per query calculations
        var qSquaredNorm = VectorUtil.dotProduct(q, q);
        var qTB = basis.project(q);
        switch (similarityFunction) {
            case EUCLIDEAN:
                // return function that computes similarity to query
                return new NodeSimilarity.EstimatedNeighborsScoreFunction() {
                    float[] c;
                    float cSquaredNorm;
                    float t;
                    float[] dProjScalarFactors;
                    float[] dResSquaredComponents;
                    float[] dResNorms;
                    float qResSquaredNorm;
                    double qResNorm;
                    long[] sgnDResTBs;
                    float[] cTB;
                    long sqnqResidualProjection;
                    @Override
                    public void swapBaseNode(int node2, float dotProduct) {
                        c = ravv.vectorValue(node2);
                        cSquaredNorm = cSquaredNorms[node2];
                        t = dotProduct / cSquaredNorm;
                        dProjScalarFactors = dProjScalarFactor[node2];
                        dResSquaredComponents = dResSquared[node2];
                        dResNorms = dRes[node2];
                        qResSquaredNorm = qSquaredNorm - (t * t * cSquaredNorm);
                        qResNorm = Math.sqrt(qResSquaredNorm);
                        sgnDResTBs = sgnDResTB[node2];
                        cTB = cBasisProjections[node2];
                        // L2 distance squared is
                        // qproj - dproj L2 distance squared plus // DONE
                        // qres L2 norm squared plus // DONE
                        // dres L2 norm squared minus // DONE
                        // 2qtresdres // DONE
                        sqnqResidualProjection = 0L; // assuming low-rank 64
                        for (int k = 0; k < 64; k++) {
                            if ( qTB[k] - t * cTB[k] >= 0) {
                                sqnqResidualProjection |= 1L << k;
                            }
                        }
                    }

                    @Override
                    public float similarityTo(int neighborIndex) {
                        var distance = (float) ((t - dProjScalarFactors[neighborIndex]) * (t - dProjScalarFactors[neighborIndex]) * cSquaredNorm +
                                qResSquaredNorm +
                                dResSquaredComponents[neighborIndex] -
                                2 * dResNorms[neighborIndex] * qResNorm * // just need to approximate cos(qres, dres)
                                        cachedCosine[Long.bitCount(sqnqResidualProjection ^ sgnDResTBs[neighborIndex])]);
                        return 1 / ( 1 + distance);
                    }

                    @Override
                    public float[] bulkSimilarityTo(int node2, float topScore, long nodesToInclude) {
                        //node2 is our c index
                        var c = ravv.vectorValue(node2);
                        var cSquaredNorm = cSquaredNorms[node2];
                        var t = VectorUtil.dotProduct(q,c) / cSquaredNorm; // UPDATE TO USE CACHED PREVIOUS DISTANCE
                        var dProjScalarFactors = dProjScalarFactor[node2];
                        var dResSquaredComponents = dResSquared[node2];
                        var dResNorms = dRes[node2];
                        var qResSquaredNorm = qSquaredNorm - (t * t * cSquaredNorm);
                        var qResNorm = Math.sqrt(qResSquaredNorm);
                        var sgnDResTBs = sgnDResTB[node2];
                        var cTB = cBasisProjections[node2];

                        // L2 distance squared is
                        // qproj - dproj L2 distance squared plus // DONE
                        // qres L2 norm squared plus // DONE
                        // dres L2 norm squared minus // DONE
                        // 2qtresdres // DONE
                        sqnqResidualProjection = 0L; // assuming low-rank 64
                        for (int k = 0; k < 64; k++) {
                            if ( qTB[k] - t * cTB[k] >= 0) {
                                sqnqResidualProjection |= 1L << k;
                            }
                        }

                        float[] results = new float[dProjScalarFactors.length];
                        for (int i = 0; i < results.length; i++) {
                            var distance = (float) ((t - dProjScalarFactors[i]) * (t - dProjScalarFactors[i]) * cSquaredNorm +
                                    qResSquaredNorm +
                                    dResSquaredComponents[i] -
                                    2 * dResNorms[i] * qResNorm * // just need to approximate cos(qres, dres)
                                            cachedCosine[Long.bitCount(sqnqResidualProjection ^ sgnDResTBs[i])]);
                            results[i] = 1 / (1 + distance);
                        }
                        return results;
                    }

                    @Override
                    public Map<Integer, Float> getDotProductCache() {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public float getQSquaredNorm() {
                        return qSquaredNorm;
                    }

                    @Override
                    public float[] getQTB() {
                        return qTB;
                    }
                };
            case DOT_PRODUCT:
                return new NodeSimilarity.EstimatedNeighborsScoreFunction() {
                    float[] c;
                    float cSquaredNorm;
                    float t;
                    float[] dProjScalarFactors;
                    float[] dResNorms;
                    float qResSquaredNorm;
                    double qResNorm;
                    long[] sgnDResTBs;
                    float[] cTB;
                    long sqnqResidualProjection;
                    float tcSquaredNorm;
                    public final HashMap<Integer, Float> dotProductCache = new HashMap<>();
                    @Override
                    public void swapBaseNode(int node2, float dotProduct) {
                        c = ravv.vectorValue(node2);
                        cSquaredNorm = cSquaredNorms[node2];
                        t = dotProduct / cSquaredNorm;
                        dProjScalarFactors = dProjScalarFactor[node2];
                        dResNorms = dRes[node2];
                        tcSquaredNorm = t * cSquaredNorm;
                        qResSquaredNorm = qSquaredNorm - (t * tcSquaredNorm);
                        qResNorm = Math.sqrt(qResSquaredNorm);
                        sgnDResTBs = sgnDResTB[node2];
                        cTB = cBasisProjections[node2];
                        sqnqResidualProjection = VectorUtil.matrixDifferenceSigns(qTB, cTB, t);
                    }

                    @Override
                    public float similarityTo(int neighborIndex) {
                        // qprojt (dot) dproj + qrest (dot) dres
                        // qproj is t * c, dproj is bc, pull factors out, cSquaredNorm * t * b
                        var temp2 = t * cSquaredNorm * dProjScalarFactors[neighborIndex];
                        // qrest (dot) dres follows
                        var temp = dResNorms[neighborIndex] * qResNorm * // just need to approximate cos(qres, dres)
                                cachedCosine[Long.bitCount(sqnqResidualProjection ^ sgnDResTBs[neighborIndex])];
                        return (float) (temp2 + temp);
                    }

                    @Override
                    public float[] bulkSimilarityTo(int node2, float dotProduct, long nodesToInclude) {
                        return VectorUtil.fingerDotProduct(FingerMetadata.this, this, node2, dotProduct,
                                nodesToInclude);
                    }

                    @Override
                    public Map<Integer, Float> getDotProductCache() {
                        return dotProductCache;
                    }

                    @Override
                    public float getQSquaredNorm() {
                        return qSquaredNorm;
                    }

                    @Override
                    public float[] getQTB() {
                        return qTB;
                    }
                };
            default:
                throw new IllegalArgumentException("Unsupported similarity function " + similarityFunction);
        }
    }



}
