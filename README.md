# JVector
<a href="https://trendshift.io/repositories/2946" target="_blank"><img src="https://trendshift.io/api/badge/repositories/2946" alt="jbellis%2Fjvector | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

JVector is a pure Java embedded vector search engine, used by [DataStax Astra DB](https://www.datastax.com/products/datastax-astra) and (soon) Apache Cassandra.

What is JVector?
- Algorithmic-fast. JVector uses state of the art graph algorithms inspired by DiskANN and related research that offer high recall and low latency.
- Implementation-fast. JVector uses the Panama SIMD API to accelerate index build and queries.
- Memory efficient. JVector compresses vectors using product quantization so they can stay in memory during searches.  (As part of our PQ implementation, our SIMD-accelerated kmeans class is 5x faster than the one in Apache Commons Math.)
- Disk-aware. JVector’s disk layout is designed to do the minimum necessary iops at query time.
- Concurrent.  Index builds scale linearly to at least 32 threads.  Double the threads, half the build time.
- Incremental. Query your index as you build it.  No delay between adding a vector and being able to find it in search results.
- Easy to embed. API designed for easy embedding, by people using it in production.


## JVector performance, visualized
JVector vs Lucene searching the Deep100M dataset (about 35GB of vectors and 25GB index):
![Screenshot from 2023-09-29 16-39-33](https://github.com/jbellis/jvector/assets/42158/7710f33d-ff6a-4282-9e31-4a5eaacd796f)

JVector scales updates linearly to at least 32 threads:
![Screenshot from 2023-09-14 18-05-15](https://github.com/jbellis/jvector/assets/42158/f0127bfc-6c45-48b9-96ea-95b2120da0d9)

## Upgrading from 1.0.x to 2.0.x

See [UPGRADING.md](./UPGRADING.md).

## JVector basics
Adding to your project. Replace `${latest-version}` with ![Maven Central](https://img.shields.io/maven-central/v/io.github.jbellis/jvector?color=green). Example `<version>1.0.1</version>`:

```
<dependency>        
    <groupId>io.github.jbellis</groupId>          
    <artifactId>jvector</artifactId>
    <!-- Use the latest version from https://central.sonatype.com/artifact/io.github.jbellis/jvector -->
    <version>${latest-version}</version>
</dependency>
```

Building the index:
- [`GraphIndexBuilder`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/GraphIndexBuilder.java) is the entry point for building a graph.  You will need to implement
  [`RandomAccessVectorValues`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/RandomAccessVectorValues.java) to provide vectors to the builder;
  [`ListRandomAccessVectorValues`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/ListRandomAccessVectorValues.java) is a good starting point.
- If all your vectors
  are in the provider
  up front, you can just call `build()` and it will parallelize the build across
  all available cores.  Otherwise you can call `addGraphNode` as you add vectors; 
  this is non-blocking and can be called concurrently from multiple threads.
- Call `GraphIndexBuilder.cleanup` when you are done adding vectors.  This will
  optimize the index and make it ready to write to disk.  (Graphs that are
  in the process of being built can be searched at any time; you do not have to call
  `cleanup` first.)

Searching the index:
- [`GraphSearcher`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/GraphSearcher.java) is the entry point for searching.  Results come back as a [`SearchResult`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/SearchResult.java) object that contains node IDs and scores, in
  descending order of similarity to the query vector.  `GraphSearcher` objects are re-usable,
  so unless you have a very simple use case you should use `GraphSearcher.Builder` to
  create them; `GraphSearcher.search` is also available with simple defaults, but calling it
  will instantiate a new `GraphSearcher` every time so performance will be worse.
- JVector represents vectors in the index as the ordinal (int) corresponding to their
  index in the `RandomAccessVectorValues` you provided.  You can get the original vector
  back with `GraphIndex.getVector`, if necessary, but since this is a disk-backed index
  you should design your application to avoid doing so unnecessarily.

## DiskANN and Product Quantization 
JVector implements [DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf)-style 
search, meaning that vectors can be compressed using product quantization so that searches
can be performed using the compressed representation that is kept in memory.  You can enable
this with the following steps:
- Create a [`VectorCompressor`](./jvector-base/src/main/java/io/github/jbellis/jvector/pq/VectorCompressor.java) object with your vectors using either `ProductQuantization.compute`
- or `BinaryQuantization.compute`.  PQ is more flexible than BQ and is less lossy: even at the same compressed size, 
  in the datasets tested by Bench, only the ada002 vectors in the wikipedia dataset
  are large enough and/or overparameterized enough to benefit from BQ while achieving recall
  competitive with PQ.  However, if you are dealing with very large vectors and/or your
  recall requirement is not strict, you may still want to try BQ since it is MUCH faster to both compute and search with.
- Use `VectorCompressor::encode` or `encodeAll` to encode your vectors, then call
  `VectorCompressor::createCompressedVectors` to create a `CompressedVectors` object.
- Call `CompressedVectors::approximateScoreFunctionFor` to create a [`NeighborSimilarity.ApproximateScoreFunction`](./jvector-base/src/main/java/io/github/jbellis/jvector/graph/NeighborSimilarity.java) for your query that uses the
  compressed vectors to accelerate search, and pass this
  to the `GraphSearcher.search` method.

## Saving and loading indexes
- [`OnDiskGraphIndex`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/OnDiskGraphIndex.java) and [`CompressedVectors`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/CompressedVectors.java) have `write()` methods to save state to disk.
  They initialize from disk using their constructor and `load()` methods, respectively.
  Writing just requires a DataOutput, but reading requires an 
  implementation of [`RandomAccessReader`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/RandomAccessReader.java) and the related `ReaderSupplier` to wrap your
  preferred i/o class for best performance. See `SimpleMappedReader` and `SimpleMappedReaderSupplier` for an example.
- Building a graph does not technically require your RandomAccessVectorValues object
  to live in memory, but it will perform much better if it does.  `OnDiskGraphIndex`,
  by contrast, is designed to live on disk and use minimal memory otherwise.
- You can optionally wrap `OnDiskGraphIndex` in a [`CachingGraphIndex`](./jvector-base/src/main/java/io/github/jbellis/jvector/disk/CachingGraphIndex.java) to keep the most commonly accessed
  nodes (the ones nearest to the graph entry point) in memory.

## Advanced configuration

- JVector heavily utilizes the Panama Vector API(SIMD) for ANN indexing and search.  We have seen cases where the memory 
bandwidth is saturated during indexing and product quantization and can cause the process to slow down. To avoid 
this, index and PQ builds use a [`PhysicalCoreExecutor`](./jvector-base/src/main/java/io/github/jbellis/jvector/util/PhysicalCoreExecutor.java) 
to limit the amount of operations to the physical core count. The default value is 1/2 the processor count seen by Java.
This may not be correct in all setups (e.g. no hyperthreading or hybrid architectures) so if you wish to override the default use the `-Djvector.physical_core_count` property. 

## Sample code
- The [`SiftSmall`](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/SiftSmall.java) class demonstrates how to put all of the above together to index and search the
  "small" SIFT dataset of 10,000 vectors.
- The [`Bench`](./jvector-examples/src/main/java/io/github/jbellis/jvector/example/Bench.java) class performs grid search across the `GraphIndexBuilder` parameter space to find
  the best tradeoffs between recall and throughput.  You can use [`plot_output.py`](./plot_output.py) to graph the [pareto-optimal
  points](https://en.wikipedia.org/wiki/Pareto_efficiency) found by `Bench`.

Some sample KNN datasets for testing based on ada-002 embeddings generated on wikipedia data are available in ivec/fvec format for testing at:

```
aws s3 ls s3://astra-vector/wikipedia_scout/ --no-sign-request 
                           PRE 100k/
                           PRE 1M/
                           PRE 4M/
```

Bench (see below) automatically downloads the 100k dataset to the `./fvec` directory

## Developing and Testing
This project is organized as a [multimodule Maven build](https://maven.apache.org/guides/mini/guide-multiple-modules.html). The intent is to produce a multirelease jar suitable for use as
a dependency from any Java 11 code. When run on a Java 20+ JVM with the Vector module enabled, optimized vector 
providers will be used. In general, the project is structured to be built with JDK 20+, but when `JAVA_HOME` is set to
Java 11 -> Java 19, certain build features will still be available.

Base code is in [jvector-base](./jvector-base) and will be built for Java 11 releases, restricting language features and APIs
appropriately. Code in [jvector-twenty](./jvector-twenty) will be compiled for Java 20 language features/APIs and included in the final
multirelease jar targeting supported JVMs. [jvector-multirelease](./jvector-multirelease) packages [jvector-base](./jvector-base) and [jvector-twenty](./jvector-twenty) as a
multirelease jar for release. [jvector-examples](./jvector-examples) is an additional sibling module that uses the reactor-representation of
jvector-base/jvector-twenty to run example code. [jvector-tests](./jvector-tests) contains tests for the project, capable of running against 
both Java 11 and Java 20+ JVMs.

To run tests, use `mvn test`. To run tests against Java 20+, use `mvn test`. To run tests against Java 11, use `mvn -Pjdk11 test`. To run a single test class, 
use the Maven Surefire test filtering capability, e.g., `mvn -Dtest=TestNeighborArray test`. You may also use method-level filtering and patterns, e.g.,
`mvn -Dtest=TestNeighborArray#testRetain* test`.

You can run `SiftSmall` and `Bench` directly to get an idea of what all is going on here. `Bench` will automatically download required datasets to the `fvec` and `hdf5` directories. 
The files used by `SiftSmall` can be found in the [siftsmall directory](./siftsmall) in the project root.

To run either class, you can use the Maven exec-plugin via the following incantations:

> `mvn compile exec:exec@bench`

or for Sift:

> `mvn compile exec:exec@sift`

`Bench` takes an optional `benchArgs` argument that can be set to a list of whitespace-separated regexes. If any of the
provided regexes match within a dataset name, that dataset will be included in the benchmark. For example, to run only the glove
and nytimes datasets, you could use:

> `mvn compile exec:exec@bench -DbenchArgs="glove nytimes"`

To run Sift/Bench without the JVM vector module available, you can use the following invocations:

> `mvn -Pjdk11 compile exec:exec@bench`

> `mvn -Pjdk11 compile exec:exec@sift`

The `... -Pjdk11` invocations will also work with `JAVA_HOME` pointing at a Java 11 installation.

To release, configure `~/.m2/settings.xml` to point to OSSRH and run `mvn -Prelease clean deploy`.

---
