#echo "--------cuda nearest neighbor------------------------"
#for ((count=1;count<=30;count++)) do
#   echo -n -e "$count.\t" 
#   ./benchmarks/cuda/NN_new2 900000000
#done
#echo "--------ske_lib nearest neighbor------------------------"
#for ((count=1;count<=30;count++)) do
#   echo -n -e "$count.\t" 
#   mix run benchmarks/ske_lib/nearest_neighbor.ex 900000000
#done
echo "--------cuda saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/cuda/SAXPY 900000000
done
echo "--------ske_lib saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/ske_lib/saxpy_rts.ex 900000000
done
