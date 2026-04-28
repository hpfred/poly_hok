echo "--------cuda dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/DP_new 900000000
done
echo "--------ske_lib dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/dot_product.ex 900000000
done
echo "--------cuda julia------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/JL_new 23170
done
echo "--------ske_lib julia------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/julia.ex 23170
done
echo "--------cuda nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/NB_new 500000
done
echo "--------ske_lib nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/nbodies.ex 500000
done
echo "--------cuda raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/RT_new 23170
done
echo "--------ske_lib raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/raytracer.ex 23170
done
