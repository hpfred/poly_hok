require PolyHok
require Integer
#Nx.default_backend(EXLA.Backend)
#import Nx

PolyHok.defmodule DP do
  include CAS

  def replicate(n, x), do: (for _ <- 1..n, do: x)
  def rep_change(n,x), do: rep_pos(n,x)
  def rep_pos(0,_x), do: []
  def rep_pos(n,x), do:  [x | rep_neg(n-1,x)]
  def rep_neg(0,_x), do: []
  def rep_neg(n,x), do:  [-x | rep_pos(n-1,x)]
  def new_dataset_nx_a(n), do: gen_nx_f(n,a_gen_new_dataset_nx_f(div(n,2),<<>>,<<>>))
  defp a_gen_new_dataset_nx_f(0,a1,a2), do: <<a1::binary,a2::binary>>

  defp a_gen_new_dataset_nx_f(size, a1,a2) do
    {ax,ay} = if (rem(size,2) == 0) do
                v = :rand.uniform(100)/1
                {v,-v}
              else
                v = :rand.uniform(100)/1
                {-v,v}
              end

    a_gen_new_dataset_nx_f(
        size - 1,
        <<a1::binary, ax::float-little-32>>,
        <<a2::binary, ay::float-little-32>>
    )
  end

  def new_dataset_nx_b(n), do: gen_nx_f(n,b_gen_new_dataset_nx_f(div(n,2),<<>>,<<>>))
  defp b_gen_new_dataset_nx_f(0,b1,b2), do: <<b1::binary,b2::binary>>

  defp b_gen_new_dataset_nx_f(size, b1,b2) do
    b = :rand.uniform(5)/1

    b_gen_new_dataset_nx_f(
        size - 1,
        <<b1::binary, b::float-little-32>>,
        <<b2::binary, b::float-little-32>>
    )
  end

  defp gen_nx_f(size,ref), do:  %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: {:f,32}, shape: {1,size}, names: [nil,nil]}
end

#PolyHok.include [DP]
use Ske

start = System.monotonic_time()

[arg] = System.argv()

n = String.to_integer(arg)

:rand.seed(:exsss, {123, 123, 123})

#vet1 = DP.new_dataset_nx_a(n)
vet1 = Nx.tensor(Enum.to_list(1..(n)), type: :f32)
#vet2 = DP.new_dataset_nx_b(n)
vet2 = Nx.tensor(Enum.to_list(n+1..(n+n)), type: :f32)

#IO.inspect(vet1)
#IO.inspect(vet2)

prev = System.monotonic_time()

ref1 = PolyHok.new_gnx(vet1)
#ref1 = PolyHok.new_nx_from_function(m,m,{:f,32},fn -> :rand.uniform(1000) end )
ref2 = PolyHok.new_gnx(vet2)
#IO.inspect(PolyHok.get_gnx(ref1))
#IO.inspect(PolyHok.get_gnx(ref2))

_result = ref1
    |> Ske.map(ref2, PolyHok.phok fn (a,b) -> a * b end)
#IO.inspect(PolyHok.get_gnx(result))
#_result = result
    |> Ske.reduce(0.0,PolyHok.phok fn (a,b) -> a + b end)
    |> PolyHok.get_gnx
    #|> IO.inspect

next = System.monotonic_time()

IO.puts "PolyHok\t#{n}\tTotal: #{System.convert_time_unit(next-start,:native,:millisecond)}\tGPU: #{System.convert_time_unit(next-prev,:native,:millisecond)}"
