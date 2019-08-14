
add_library(benchmark_core SHARED core.hpp core.cpp)
target_link_libraries(benchmark_core PUBLIC lbfgs ${BLAS_LIBRARIES})

include(ExternalProject)

find_package(Threads REQUIRED)
ExternalProject_Add(google_benchmark_build
  GIT_REPOSITORY    https://github.com/google/benchmark.git
  GIT_TAG           v1.5.0
  CMAKE_ARGS        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF
                    -DBENCHMARK_ENABLE_INSTALL=OFF -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
  INSTALL_COMMAND   ""
  BUILD_BYPRODUCTS  "google_benchmark_build-prefix/src/google_benchmark_build-build/src/libbenchmark.a"
                    "google_benchmark_build-prefix/src/google_benchmark_build-build/src/libbenchmark_main.a"
)
add_library(Benchmark INTERFACE)
add_dependencies(Benchmark google_benchmark_build)
ExternalProject_Get_Property(google_benchmark_build BINARY_DIR)
target_include_directories(Benchmark SYSTEM INTERFACE ${BINARY_DIR}/include)
target_link_libraries(Benchmark INTERFACE ${BINARY_DIR}/src/libbenchmark_main.a
                                          ${BINARY_DIR}/src/libbenchmark.a)
target_link_libraries(Benchmark INTERFACE Threads::Threads)


add_executable(main main.cpp)
target_link_libraries(main PUBLIC benchmark_core Benchmark)
