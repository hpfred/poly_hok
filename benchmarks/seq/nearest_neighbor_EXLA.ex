defmodule DataSet do
  def gen_data_set(n) do
    key = Nx.Random.key(123)
    {lat, _key} = Nx.Random.uniform(key, shape: {n}, type: :f32)
    {lon, _key2} = Nx.Random.uniform(Nx.Random.key(456), shape: {n}, type: :f32)

    lat = Nx.add(Nx.multiply(lat, Nx.tensor(63.0, type: :f32)), Nx.tensor(7.0, type: :f32))
    lon = Nx.multiply(lon, Nx.tensor(359.0, type: :f32))

    Nx.stack([lat, lon], axis: 1)
  end
end

arg = hd(System.argv())
size = String.to_integer(arg)

# Escolha um backend antes de criar os tensores:
Nx.default_backend({EXLA.Backend, []})
#Nx.default_backend({Torchx.Backend, []})

points = DataSet.gen_data_set(size)

prev = System.monotonic_time()

dist_sq = Nx.sum(Nx.multiply(points, points), axes: [1])
min_dist = Nx.sqrt(Nx.reduce_min(dist_sq))

next = System.monotonic_time()
IO.puts("Nx\t#{size}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
