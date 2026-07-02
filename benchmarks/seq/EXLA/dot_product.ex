arg = hd(System.argv())

n = String.to_integer(arg)

## EXLA Backend
Nx.default_backend({EXLA.Backend, client: :host})
#Nx.Defn.default_options(compiler: EXLA, client: :host)
## PYTORCH Backend
#Nx.default_backend({Torchx.Backend, []})

# Cria tensores NX com os valores 1..n
x = Nx.tensor(Enum.to_list(1..n), type: :f32)
y = Nx.tensor(Enum.to_list(1..n), type: :f32)

prev = System.monotonic_time()

# dot product com NX
_result = Nx.dot(x, y)
# Alternativa explícita:
# result = Nx.sum(Nx.multiply(x, y))

next = System.monotonic_time()

IO.puts("Nx\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
