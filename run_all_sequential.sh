echo "--------default dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/seq/dot_product.ex 200000000
done
echo "--------exla dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/dot_product_EXLA.ex 200000000
done
echo "--------pytorch dot product------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/dot_product_pytorch.ex 200000000
done

#echo "--------julia------------------------"
#Not Implemented

echo "--------default nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/seq/nbodies.ex 5000
done
echo "--------exla nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/nbodies_EXLA.ex 5000
done
echo "--------pytorch nbodies------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/nbodies_pytoch.ex 5000
done

echo "--------default raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/seq/raytracer.ex 2160
done
echo "--------exla raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/raytracer_EXLA.ex 2160
done
echo "--------pytorch raytracer------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/raytracer_pytorch.ex 2160
done

##echo "--------matrix multiplication-------------------"
##Not implemented

echo "--------default nearest neighbor------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/seq/nearest_neighbor.ex 100000000
done
echo "--------exla nearest neighbor------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/nearest_neighbor_EXLA.ex 100000000
done
echo "--------pytorch nearest neighbor------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/nearest_neighbor_pytorch.ex 100000000
done

echo "--------default saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   ./benchmarks/seq/saxpy.ex 200000000
done
echo "--------exla saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/saxpy_EXLA.ex 200000000
done
echo "--------pytorch saxpy------------------------"
for ((count=1;count<=30;count++)) do
   echo -n -e "$count.\t" 
   mix run benchmarks/seq/saxpy_pytorch.ex 200000000
done
