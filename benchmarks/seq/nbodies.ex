defmodule NBody do
  @softening 1.0e-9
  @dt 0.01

  def gen_bodies(n) do
    {positions, key} = Nx.Random.uniform(Nx.Random.key(123), shape: {n, 3}, type: :f32)
    {velocities, _key2} = Nx.Random.uniform(key, shape: {n, 3}, type: :f32)

    {positions, Nx.multiply(velocities, 0.01)}
  end

  def compute_forces(positions) do
    n = elem(Nx.shape(positions), 0)

    positions_i = Nx.reshape(positions, {n, 1, 3})
    positions_j = Nx.reshape(positions, {1, n, 3})
    diff = Nx.subtract(positions_j, positions_i)

    dist_sq = Nx.sum(Nx.multiply(diff, diff), axes: [2])
    dist_sq = Nx.add(dist_sq, @softening)
    inv_dist = Nx.rsqrt(dist_sq)
    inv_dist3 = Nx.multiply(Nx.multiply(inv_dist, inv_dist), inv_dist)

    Nx.sum(Nx.multiply(diff, Nx.reshape(inv_dist3, {n, n, 1})), axes: [1])
  end

  def step(positions, velocities) do
    forces = compute_forces(positions)
    velocities = Nx.add(velocities, Nx.multiply(forces, @dt))
    positions = Nx.add(positions, Nx.multiply(velocities, @dt))
    {positions, velocities}
  end
end

[arg] = System.argv()
n_bodies = String.to_integer(arg)

{positions, velocities} = NBody.gen_bodies(n_bodies)

prev = System.monotonic_time()

{_positions, _velocities} = NBody.step(positions, velocities)

next = System.monotonic_time()
IO.puts("Nx\t#{n_bodies}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
