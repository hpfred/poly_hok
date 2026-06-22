arg = hd(System.argv())

n = String.to_integer(arg)

x = Nx.tensor(Enum.to_list(1..n), type: :f32)
y = Nx.tensor(Enum.to_list(1..n), type: :f32)

prev = System.monotonic_time()

result = Nx.add(Nx.multiply(x, 2.0), y)

next = System.monotonic_time()
IO.puts("Nx\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
