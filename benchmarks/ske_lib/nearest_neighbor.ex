require Integer
require PolyHok

defmodule DataSet do
    def open_data_set(file) do
        {:ok, contents} = File.read(file)
        contents
        |> String.split("\n", trim: true)
        |> Enum.map(fn f ->  load_file(f) end)
        |> Enum.concat()
        |> Enum.concat()
        #|> Enum.unzip()
    end
    def load_file(file) do
        #IO.puts file
        {:ok, contents} = File.read(file)
        contents
        |> String.split("\n", trim: true)
        |> Enum.map(fn line -> words = String.split(line, " ", trim: true)
                            [ elem(Float.parse(Enum.at(words, 6)),0), elem(Float.parse(Enum.at(words,7)), 0) ] end  )
    end
    def gen_data_set_nx_double(n) do
        lat = (7 + Enum.random(0..63)) + :rand.uniform()
        lon = (Enum.random(0..358)) + :rand.uniform()
        acc = <<lat::float-little-64, lon::float-little-64>>
        ref = gen_bin_data_double(n-1, acc)
        %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: {:f,64}, shape: {n,2}, names:  [nil,nil]}
    end
    defp gen_bin_data_double(0, accumulator), do: accumulator
    defp gen_bin_data_double(size, accumulator) do
        lat = (7 + Enum.random(0..63)) + :rand.uniform()
        lon = (Enum.random(0..358)) + :rand.uniform()
        gen_bin_data_double(
            size - 1,
            <<accumulator::binary, lat::float-little-64, lon::float-little-64>>
        )
    end
    def gen_data_set_nx(n) do
        lat = (7 + Enum.random(0..63)) + :rand.uniform()
        lon = (Enum.random(0..358)) + :rand.uniform()
        acc = <<lat::float-little-32, lon::float-little-32>>
        ref = gen_bin_data(n-1, acc)
        %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: {:f,32}, shape: {n,2}, names:  [nil,nil]}
    end
    defp gen_bin_data(0, accumulator), do: accumulator
    defp gen_bin_data(size, accumulator) do
        lat = (7 + Enum.random(0..63)) + :rand.uniform()
        lon = (Enum.random(0..358)) + :rand.uniform()
        gen_bin_data(
            size - 1,
            <<accumulator::binary, lat::float-little-32, lon::float-little-32>>
        )
    end
    def gen_data_set(n), do: gen_data_set_(n,[])
    def gen_data_set_(0,data), do: data
    def gen_data_set_(n,data) do
        lat = (7 + Enum.random(0..63)) + :rand.uniform();
        lon = (Enum.random(0..358)) + :rand.uniform();
        gen_data_set_(n-1, [lat,lon|data])
    end

    def gen_lat_long(_l,c) do
        if(Integer.is_even(c)) do
            (Enum.random(0..358)) + :rand.uniform()
        else
            (7 + Enum.random(0..63)) + :rand.uniform()
    end
  end
end


PolyHok.defmodule NN do
    include CAS_Double
    def euclid_seq(l,lat,lng), do: euclid_seq_(l,lat,lng,[])
    def euclid_seq_([m_lat,m_lng|array],lat,lng,data) do
        # m_lat = Enum.at(array,0)
        #m_lng = Enum.at(array,1)

        value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
        #value = :math.sqrt((lat-m_lat)*(lat-m_lat)+(lng-m_lng)*(lng-m_lng))
        euclid_seq_(array,lat,lng,[value|data])
    end
    def euclid_seq_([],_lat,_lng, data) do
        data
    end

    defd euclid(d_result, d_locations, lat, lng) do
        #printf("%f  %f  %f  %f  \\n",d_locations[0], d_locations[1], lat, lng)
        d_result[0] = sqrt((lat-d_locations[0])*(lat-d_locations[0])+(lng-d_locations[1])*(lng-d_locations[1]))
    end

    defd menor(x,y) do
        if (x<y) do
            x
        else
            y
        end
    end
end

use Ske

[arg] = System.argv()

size = String.to_integer(arg)

:rand.seed(:exsss, {123, 123, 123})

data_set_host = DataSet.gen_data_set_nx_double(size)
#data_set_host = Nx.tensor(DataSet.gen_data_set(size),  type: {:f,32} )

#IO.inspect data_set_host

prev = System.monotonic_time()

d_array = PolyHok.new_gnx(data_set_host)
#IO.inspect(PolyHok.get_gnx(d_array))
type = PolyHok.get_type_gnx(d_array)
r = PolyHok.new_gnx(1,size, type)
    |> Ske.map(d_array, &NN.euclid/3, [0.0, 0.0], [return: false, dim: :one, coord: false])
IO.inspect(PolyHok.get_gnx(r))
_r = r
    |> Ske.reduce(50000.0,&NN.menor/2)
    |> PolyHok.get_gnx
    |> IO.inspect

next = System.monotonic_time()
IO.puts "PolyHok\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

#result_elixir = Enum.reverse(NN.euclid_seq(list_data_set,0.0,0.0))

#IO.puts("NN = #{nn[1]}")

#IO.inspect (Enum.reduce(result_elixir,0, fn (x,y)-> if y == 0 do x else if x<y do x else y end end end))
