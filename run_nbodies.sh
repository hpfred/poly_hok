echo "--------cuda nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/NB 500000
done
echo "--------ske_lib nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/nbodies.ex 500000
done
