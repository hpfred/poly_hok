defmodule Raytracer do
  @moduledoc false

  def gen_spheres(count, dim, max_radius, min_z) do
    key = Nx.Random.key(123)

    {x, key} = Nx.Random.uniform(key, shape: {count}, type: :f32)
    {y, key} = Nx.Random.uniform(key, shape: {count}, type: :f32)
    {z, key} = Nx.Random.uniform(key, shape: {count}, type: :f32)
    {radius, key} = Nx.Random.uniform(key, shape: {count}, type: :f32)
    {colors, _key} = Nx.Random.uniform(key, shape: {count, 3}, type: :f32)

    half = Nx.tensor(dim / 2, type: :f32)
    x = Nx.subtract(Nx.multiply(x, Nx.tensor(dim, type: :f32)), half)
    y = Nx.subtract(Nx.multiply(y, Nx.tensor(dim, type: :f32)), half)
    z = Nx.subtract(Nx.multiply(z, Nx.tensor(256.0, type: :f32)), Nx.tensor(128.0, type: :f32))
    radius = Nx.add(Nx.multiply(radius, Nx.tensor(max_radius, type: :f32)), Nx.tensor(min_z, type: :f32))

    %{x: x, y: y, z: z, radius: radius, colors: colors}
  end

  def raytrace(dim, spheres) do
    half = Nx.tensor(dim / 2, type: :f32)
    x = Nx.subtract(Nx.iota({dim}, type: :f32), half)
    y = Nx.subtract(Nx.iota({dim}, type: :f32), half)

    x = Nx.reshape(x, {1, dim, 1})
    y = Nx.reshape(y, {dim, 1, 1})

    sx = Nx.reshape(spheres.x, {1, 1, elem(Nx.shape(spheres.x), 0)})
    sy = Nx.reshape(spheres.y, {1, 1, elem(Nx.shape(spheres.y), 0)})
    sz = Nx.reshape(spheres.z, {1, 1, elem(Nx.shape(spheres.z), 0)})
    radius = Nx.reshape(spheres.radius, {1, 1, elem(Nx.shape(spheres.radius), 0)})

    dx = Nx.subtract(x, sx)
    dy = Nx.subtract(y, sy)
    dist_sq = Nx.add(Nx.multiply(dx, dx), Nx.multiply(dy, dy))
    radius_sq = Nx.multiply(radius, radius)

    inside = Nx.less(dist_sq, radius_sq)
    dz = Nx.sqrt(Nx.max(Nx.subtract(radius_sq, dist_sq), Nx.tensor(0.0, type: :f32)))
    t = Nx.select(inside, Nx.add(dz, sz), Nx.broadcast(Nx.tensor(-1.0, type: :f32), {dim, dim, elem(Nx.shape(radius), 0)}))

    best = Nx.argmax(t, axis: 2)
    colors = Nx.take(spheres.colors, best)

    mask = Nx.reshape(
      Nx.reduce(inside, Nx.tensor(0, type: {:u, 8}), [axes: [2]], fn i, acc -> Nx.logical_or(i, acc) end),
      {dim, dim, 1}
    )

    mask = Nx.broadcast(mask, {dim, dim, 3})
    color = Nx.select(mask, Nx.multiply(colors, Nx.tensor(255.0, type: :f32)), Nx.broadcast(Nx.tensor(0.0, type: :f32), {dim, dim, 3}))
    alpha = Nx.broadcast(Nx.tensor(255.0, type: :f32), {dim, dim, 1})

    Nx.round(Nx.concatenate([color, alpha], axis: 2))
  end
end

[arg] = System.argv()

dim = String.to_integer(arg)

# Escolha um backend antes de criar os tensores:
#Nx.default_backend({EXLA.Backend, []})
Nx.default_backend({Torchx.Backend, []})

spheres = Raytracer.gen_spheres(20, dim, 160, 20)

prev = System.monotonic_time()

_pixels = Raytracer.raytrace(dim, spheres)

next = System.monotonic_time()
IO.puts("Nx\t#{dim}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
