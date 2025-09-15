echo "--------cuda julia------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/JL_new 11264
done
echo "--------ske_lib julia------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/julia.ex 11264
done
