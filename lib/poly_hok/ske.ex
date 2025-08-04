require PolyHok
#PolyHok.defmodule SkeKernels do
#
# defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
#    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
#    id  = step * globalId
#    #f(id,id)
#    if (globalId < size) do
#      f(d_array+id,par1,par2)
#    end
#  end
#
#end

PolyHok.defmodule Ske do
  #defmacro __using__(_opts) do
  #     IO.puts "You are USIng!"
  #    end

  include CAS_Poly

  def reduce(ref, initial, f) do
     shape = PolyHok.get_shape_gnx(ref)
     type = PolyHok.get_type_gnx(ref)
     size = Tuple.product(shape)
      result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

      threadsPerBlock = 256
      blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
      numberOfBlocks = blocksPerGrid

      case type do
        {:f,32} -> cas = PolyHok.phok (fn (x,y,z) -> cas_float(x,y,z) end)
            PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

        {:f,64} -> cas = PolyHok.phok (fn (x,y,z) -> cas_double(x,y,z) end)
            PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

        {:s,32} -> cas = PolyHok.phok (fn (x,y,z) -> cas_int(x,y,z) end)
            PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

        x -> raise "new_gnx: type #{x} not suported"
     end

      result_gpu
  end
  defk reduce_kernel(a, ref4, initial, n, cas, f) do
    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = initial

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do  ###&& tid < n) do
      #tid = blockDim.x * gridDim.x + tid
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

      __syncthreads()
      i = i/2
    end

  if (cacheIndex == 0) do
    current_value = ref4[0]
    while(!(current_value == cas(ref4,current_value,f(cache[0],current_value)))) do
      current_value = ref4[0]
    end
  end

  end

  @defaults %{coord: false, return: true, dim: :one}
  def map(a,b,c,options \\[])
  def map({:nx, type, shape, name , ref}, func, [par1], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_1para_1D({:nx, type, shape, name , ref}, par1, func)

              else if (not coord && return) do
                map_1para_1D_resp({:nx, type, shape, name , ref}, par1, func)

              else if (coord && not return) do
                map_1para_coord_1D({:nx, type, shape, name , ref},  par1, func)

              else if (coord && return) do
                map_1para_coord_1D_resp({:nx, type, shape, name , ref},  par1, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_1para_2D({:nx, type, shape, name , ref},  par1, func)

              else if (not coord && return) do
                map_1para_2D_resp({:nx, type, shape, name , ref},  par1, func)
     
              else if (coord && not return) do
                map_1para_coord_2D({:nx, type, shape, name , ref}, par1,  func)

              else if (coord && return) do
                map_1para_coord_2D_resp({:nx, type, shape, name , ref},  par1, func)
              end
              end
              end
              end

  end
  end

  def map({:nx, type, shape, name , ref}, func, [par1,par2], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_2para_1D({:nx, type, shape, name , ref},  par1, par2, func)

              else if (not coord && return) do
                map_2para_1D_resp({:nx, type, shape, name , ref},  par1, par2, func)

              else if (coord && not return) do
                map_2para_coord_1D({:nx, type, shape, name , ref},  par1, par2, func)

              else if (coord && return) do
                map_2para_coord_1D_resp({:nx, type, shape, name , ref},  par1, par2, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_2para_2D({:nx, type, shape, name , ref},  par1, par2, func)

              else if (not coord && return) do
                map_2para_2D_resp({:nx, type, shape, name , ref},  par1, par2, func)
     
              else if (coord && not return) do
                map_2para_coord_2D({:nx, type, shape, name , ref}, par1, par2, func)

              else if (coord && return) do
                map_2para_coord_2D_resp({:nx, type, shape, name , ref},  par1, par2, func)
              end
              end
              end
              end

  end
  end

  def map({:nx, type, shape, name , ref}, func, [par1,par2,par3], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_3para_1D({:nx, type, shape, name , ref}, par1, par2, par3, func)

              else if (not coord && return) do
                map_3para_1D_resp({:nx, type, shape, name , ref}, par1, par2, par3, func)

              else if (coord && not return) do
                map_3para_coord_1D({:nx, type, shape, name , ref}, par1, par2, par3, func)

              else if (coord && return) do
                map_3para_coord_1D_resp({:nx, type, shape, name , ref}, par1, par2, par3, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_3para_2D({:nx, type, shape, name , ref}, par1, par2, par3, func)

              else if (not coord && return) do
                map_3para_2D_resp({:nx, type, shape, name , ref}, par1, par2, par3, func)
     
              else if (coord && not return) do
                map_3para_coord_2D({:nx, type, shape, name , ref}, par1, par2, par3, func)

              else if (coord && return) do
                map_3para_coord_2D_resp({:nx, type, shape, name , ref}, par1, par2, par3, func)
              end
              end
              end
              end

  end
  end

  def map({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func, options) do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)

    if (coord || not return || dim == :two) do
      raise "The only options for a map2 are: coord: false, return: true, dim: :one"
    else
      map2({:nx, type, shape, name , ref}, {:nx, type2, shape2, name2, ref2}, func)
    end
  end
  defk map_ker(a1,a2,size,f) do
      index = blockIdx.x * blockDim.x + threadIdx.x
      stride = blockDim.x * gridDim.x

      for i in range(index,size,stride) do
            a2[i] = f(a1[i])
      end
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)
    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&Ske.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    #IO.inspect(result_gpu)
    result_gpu
  end
  defk map2_kernel(a1, a2, a3, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      a3[id] = f(a1[id],a2[id])
    end
  end
  def map2(t1, t2, func) do
    shape = PolyHok.get_shape_gnx(t1)
    type = PolyHok.get_type_gnx(t2)
    size = Tuple.product shape
    result_gpu = PolyHok.new_gnx(shape, type)

      threadsPerBlock = 256;
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      PolyHok.spawn(&Ske.map2_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])

      result_gpu
  end

  defk map_1para_1D_kernel(d_array, par1, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    for i in range(id,size,stride) do
      f(d_array[i], par1)
    end
  end
  def map_1para_1D(d_array, par1, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    size = l*step
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_1D_kernel/4,{nBlocks,1,1},{block_size,1,1},[d_array,par1,size,f])
  end
  defk map_1para_1D_resp_kernel(d_array, ret, par1, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    for i in range(id,size,stride) do
      ret[i] = f(d_array[i], par1)
    end
  end
  def map_1para_1D_resp(d_array, par1, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    size = l*step
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))
    
    PolyHok.spawn(&Ske.map_1para_1D_resp_kernel/5,{nBlocks,1,1},{block_size,1,1},[d_array,ret,par1,size,f])
    ret
  end
  defk map_1para_coord_1D_kernel(d_array, par1, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x
    for i in range(idX,size,stride) do
      f(d_array[i], par1,idX)
    end
  end
  def map_1para_coord_1D(d_array, par1, f) do
    ##essa parte nao tenho certeza se esta certo
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    size = sizeX*step

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_coord_1D_kernel/4,{grid_rows,1,1},{block_size,block_size,1},[d_array,par1,size,f])
  end
  defk map_1para_coord_1D_resp_kernel(d_array, ret, par1, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      ret[i] = f(d_array[i],par1,idX)
    end
  end
  def map_1para_coord_1D_resp(d_array, par1, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    size = sizeX*step

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_1para_coord_1D_resp_kernel/5,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,size,f])
    ret
  end

  defk map_1para_2D_kernel(d_array, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    if (stride < (sizeX*sizeY*step)) do
      f(d_array[stride], par1)
    end
  end
  def map_1para_2D(d_array, par1, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,step,sizeX,sizeY,f])
  end
  defk map_1para_2D_resp_kernel(d_array, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    
    if(stride < (sizeX*sizeY*step)) do
      ret[stride] = f(d_array[stride], par1)
    end
  end
  def map_1para_2D_resp(d_array, par1, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))
    
    PolyHok.spawn(&Ske.map_1para_2D_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,step,sizeX,sizeY,f])
    ret
  end
  defk map_1para_coord_2D_kernel(d_array, step, par1, sizeX, sizeY, f) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

    id  = step * offset
    if (offset < (sizeX*sizeY*step)) do
      f(d_array+id,par1,x,y)
    end
  end
  def map_1para_coord_2D(d_array, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_coord_2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,sizeX,sizeY,f])
  end
  defk map_1para_coord_2D_resp_kernel(d_array, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    if(stride < (sizeX*sizeY*step)) do
      ret[stride] = f(d_array[stride], par1, idX, idY)
    end
  end
  def map_1para_coord_2D_resp(d_array, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_1para_coord_2D_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,step,sizeX,sizeY,f])
    ret
  end

  defk map_2para_1D_kernel(d_array, step, par1, par2, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    for i in range(id,size,stride) do
      f(d_array+id,par1,par2)
    end
  end
  def map_2para_1D(d_array, par1, par2, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    size = l*step
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_1D_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,step,par1,par2,size,f])
  end
  defk map_2para_1D_resp_kernel(d_array, ret, par1, par2, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    for i in range(id,size,stride) do
      ret[i] = f(d_array[i],par1,par2)
    end
  end
  def map_2para_1D_resp(d_array, par1, par2, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    size = l*step
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_1D_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,ret,par1,par2,size,f])
    ret
  end
  defk map_2para_coord_1D_kernel(d_array, par1, par2, size, f) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    offset = x * blockDim.x * gridDim.x

    if (offset < size) do
      f(d_array[offset],par1,par2,x)
    end
  end
  def map_2para_coord_1D(d_array, par1, par2, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    size = sizeX*step
    
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_coord_1D_kernel/5,{grid_rows,1,1},{block_size,block_size,1},[d_array,par1,par2,size,f])
  end
  defk map_2para_coord_1D_resp_kernel(d_array, ret, par1, par2, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    
    #offset = idX * blockDim.x * gridDim.x
    #if(offset < size)do
    #  ret[offset] = f(d_array[offset],par1,par2,idX)
    for i in range(idX,size,stride) do
      ret[i] = f(d_array[i],par1,par2,idX)
    end
  end
  def map_2para_coord_1D_resp(d_array, par1, par2, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    size = sizeX*step

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_coord_1D_resp_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,par2,size,f])
    ret
  end
  
  defk map_2para_2D_kernel(d_array, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    if(stride < (sizeX*sizeY*step)) do
      f(d_array[stride], par1, par2)
    end
  end
  def map_2para_2D(d_array, par1, par2, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,par2,step,sizeX,sizeY,f])
  end
  defk map_2para_2D_resp_kernel(d_array, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    if(stride < (sizeX*sizeY*step)) do
      ret[stride] = f(d_array[stride], par1, par2)
    end
  end
  def map_2para_2D_resp(d_array, par1, par2, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,step,sizeX,sizeY,f])
    ret
  end
  defk map_2para_coord_2D_kernel(d_array, step, par1, par2, sizex, sizey, f) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

    id  = step * offset
    if (offset < (sizex*sizey)) do
      f(d_array+id,par1,par2,x,y)
    end
  end
  def map_2para_coord_2D(d_array, par1, par2, f) do
    {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizex + block_size - 1) / block_size)
    grid_cols = trunc ((sizey + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_coord_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,par2,sizex,sizey,f])
  end
  defk map_2para_coord_2D_resp_kernel(d_array, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    if (stride < (sizeX*sizeY*step)) do
      ret[stride] = f(d_array[stride], par1, par2, idX, idY)
      #ret[stride] = f(d_array[stride], par1, par2, threadIdx.x, threadIdx.y)
    end
  end
  def map_2para_coord_2D_resp(d_array, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_coord_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,step,sizeX,sizeY,f])
    ret
  end

  defk map_3para_1D_kernel(d_array, par1, par2, par3, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    for i in range(id,size,stride) do
      f(d_array[i],par1,par2,par3)
    end
  end
  def map_3para_1D(d_array, par1, par2, par3, f) do
      block_size =  128;
      {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      size = l*step
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_3para_1D_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,par1,par2,par3,size,f])
  end
  defk map_3para_1D_resp_kernel(d_array, ret, par1, par2, par3, size, f) do
        id = blockIdx.x * blockDim.x + threadIdx.x
        stride = blockDim.x * gridDim.x
        for i in range(id,size,stride) do
          ret[i] = f(d_array[i],par1,par2,par3)
        end
  end
  def map_3para_1D_resp(d_array, par1, par2, par3, f) do
      block_size =  128;
      {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      size = l*step
      nBlocks = floor ((size + block_size - 1) / block_size)
      ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

      PolyHok.spawn(&Ske.map_3para_1D_resp_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array,ret,par1,par2,par3,size,f])
      ret
  end
  defk map_3para_coord_1D_kernel(d_array, par1, par2, par3, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    #offset = x * blockDim.x * gridDim.x
    #if (offset < size) do
    #  f(d_array+id,par1,par2,par3,x)
    for i in range(idX,size,stride) do
      f(d_array[i],par1,par2,par3,idX)
    end
  end
  def map_3para_coord_1D(d_array, par1, par2, par3, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    size = sizeX*step

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_3para_coord_1D_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array,par1,par2,par3,size,f])
  end
  defk map_3para_coord_1D_resp_kernel(d_array, ret, par1, par2, par3, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x
    for i in range(id,size,stride) do
      ret[i] = f(d_array[i],par1,par2,par3,id)
    end
  end
  def map_3para_coord_1D_resp(d_array, par1, par2, par3, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    size = sizeX*step

    #block_size = 16
    block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    #PolyHok.spawn(&Ske.map_3para_coord_1D_resp_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,size,f])
    PolyHok.spawn(&Ske.map_3para_coord_1D_resp_kernel/7,{grid_rows,1,1},{block_size,1,1},[d_array,ret,par1,par2,par3,size,f])
    ret
  end

  defk map_3para_2D_kernel(d_array, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    if(stride < (sizeX*sizeY)) do
      f(d_array[stride], par1, par2, par3)
    end
  end
  def map_3para_2D(d_array, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_3para_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,par2,par3,step,sizeX,sizeY,f])
  end
  defk map_3para_2D_resp_kernel(d_array, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x
    if(stride < (sizeX*sizeY)) do
      ret[stride] = f(d_array[stride], par1, par2, par3)
    end
  end
  def map_3para_2D_resp(d_array, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY + block_size - 1) / block_size)
      ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

      PolyHok.spawn(&Ske.map_3para_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,step,sizeX,sizeY,f])
      ret
  end
  defk map_3para_coord_2D_kernel(d_array, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

    id  = step * offset
    if (offset < (sizeX*sizeY)) do
      f(d_array+id,par1,par2,par3,idX,idY)
    end
  end
  def map_3para_coord_2D(d_array, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_3para_coord_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,par2,par3,step,sizeX,sizeY,f])
  end
  defk map_3para_coord_2D_resp_kernel(d_array, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x


    if(stride < (sizeX*sizeY*step)) do
      ret[stride] = f(d_array[stride], par1, par2, par3, idX, idY)
    end
  end
  def map_3para_coord_2D_resp(d_array, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_3para_coord_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,step,sizeX,sizeY,f])
    ret
  end
  
end


#### kernels
