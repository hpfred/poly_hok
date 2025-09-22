require PolyHok
use Ske

PolyHok.defmodule MM do
  defd mat_mult(arr1,arr2,size,row,col) do
    arr1[row * size + col]*arr2[col * size + row]
  end
end

[arg] = System.argv()

m = String.to_integer(arg)

#vet1 = Nx.iota({m,m}, type: :f32)
#vet2 = Nx.iota({m,m}, type: :f32)

#{mat1,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {m, m}, type: :f32)
#{mat2,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {m, m}, type: :f32)

#mat1 = Matrex.new(1, m*m, fn -> :rand.uniform(1000) end)
#mat2 = Matrex.new(1, m*m, fn -> :rand.uniform(1000) end)

#mat1 = PolyHok.new_nx_from_function(m,m,{:f,32},fn -> :rand.uniform(1000) end )
#mat2 = PolyHok.new_nx_from_function(m,m,{:f,32},fn -> :rand.uniform(1000) end)

mat1 = Nx.tensor(Enum.to_list(1..(m*m)), type: :f32)
mat2 = Nx.tensor(Enum.to_list(1..(m*m)),  type: :f32)

mat1 = Nx.reshape(mat1,{m,m})
mat2 = Nx.reshape(mat2,{m,m})

prev = System.monotonic_time()


#_result = PolyHok.gpufor x <- 0..m, y <- 0..m, mat1, mat2,m do
#            sum = 0
#            for i in range(0,m,1) do
#                  sum = sum + mat1[x * m + i] * mat2[i * m + y]
#            end
#            sum
#          end

## map2_0para_coord_2D_resp
## 1 parametro na verdade, já que vai precisar saber o tamanho

arr1_gpu = PolyHok.new_gnx(mat1)
arr2_gpu = PolyHok.new_gnx(mat2)
par1 = m

result_gpu = PolyHok.new_gnx(m,m,PolyHok.get_array_type(mat1))
#MM.map2xy2D1p(arr1_gpu, arr2_gpu, par, result_gpu, size1, f)
result_gpu = Ske.map(arr1_gpu, arr2_gpu, [par1], &MM.mat_mult/5, [return: true, dim: :two, coord: true])

r_gpu = PolyHok.get_gnx(result_gpu)
r_gpu

#comp mat1 mat2 m m m(fun mat1 mat2 m x y)

next = System.monotonic_time()

IO.puts "PolyHok\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

#PolyHok.null(mat1)
#PolyHok.null(mat2)
#m1 = Matrex.reshape(mat1,m,m)
#m2 = Matrex.reshape(mat2,m,m)
#res_cpu = Matrex.dot(m1,m2)
#IO.inspect Matrex.sum(res_cpu)
#IO.inspect Matrex.sum(result)
