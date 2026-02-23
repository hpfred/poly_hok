echo "--------cuda matrix multiplication------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/MM 8192
done
echo "--------cuda matrix multiplication double------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/MMd 8192
done
echo "--------ske_lib matrix multiplication------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/mm.ex 8192
done
