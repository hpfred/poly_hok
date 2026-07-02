echo "--------cuda dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/DP_new 900000000
done
echo "--------mareco dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/dot_product.ex 900000000
done
echo "--------sequential dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/dot_product_EXLA.ex 200000000
done

echo "--------cuda nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/NB_new 500000
done
echo "--------mareco nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/nbodies.ex 500000
done
echo "--------sequential nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/nbodies_EXLA.ex 5000
done

echo "--------cuda raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/RT_new 23170
done
echo "--------mareco raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/raytracer.ex 23170
done
echo "--------sequential raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/raytracer_EXLA.ex 2160
done

echo "--------cuda nearest neighbor------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/NN_new2 900000000
done
echo "--------mareco nearest neighbor------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/nearest_neighbor.ex 900000000
done
echo "--------sequential nearest neighbor------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/nearest_neighbor_EXLA.ex 100000000
done

echo "--------cuda saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/SAXPY 900000000
done
echo "--------mareco saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/saxpy_rts.ex 900000000
done
echo "--------sequential saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/saxpy_EXLA.ex 200000000
done
