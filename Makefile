all: priv/gpu_nifs.so 

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	nvcc --shared -g -lcuda -lnvrtc --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/gpu_nifs.cu

bmp: c_src/bmp_nifs.cu 
	nvcc --shared -g --compiler-options '-fPIC' -o priv/bmp_nifs.so c_src/bmp_nifs.cu

BENCH_ALVOS := benchmarks/cuda/DP \
	       benchmarks/cuda/JL \
	       benchmarks/cuda/MM \
               benchmarks/cuda/MMd \
               benchmarks/cuda/NB \
               benchmarks/cuda/NN \
               benchmarks/cuda/NNd \
               benchmarks/cuda/RT \
               benchmarks/cuda/SAXPY \
               benchmarks/cuda/SA

.PHONY: benchmarks clean
benchmarks: $(BENCH_ALVOS)

benchmarks/cuda/DP: benchmarks/cuda/dot_product.cu
	nvcc $< -o $@
benchmarks/cuda/JL: benchmarks/cuda/julia.cu
	nvcc $< -o $@
benchmarks/cuda/MM: benchmarks/cuda/mm.cu
	nvcc $< -o $@
benchmarks/cuda/MMd: benchmarks/cuda/mm_double.cu
	nvcc $< -o $@
benchmarks/cuda/NB: benchmarks/cuda/nbodies.cu
	nvcc $< -o $@
benchmarks/cuda/NN: benchmarks/cuda/nearest_neighbor.cu
	nvcc $< -o $@
benchmarks/cuda/NNd: benchmarks/cuda/nearest_neighbor_double.cu
	nvcc $< -o $@
benchmarks/cuda/RT: benchmarks/cuda/raytracer.cu
	nvcc $< -o $@
benchmarks/cuda/SAXPY: benchmarks/cuda/saxpy.cu
	nvcc $< -o $@
benchmarks/cuda/SA: benchmarks/cuda/sum_arrays.cu
	nvcc $< -o $@

clean:
	rm -f priv/gpu_nifs.so priv/bmp_nifs.so
	find benchmarks/cuda -type f ! -name "*.*" -delete

