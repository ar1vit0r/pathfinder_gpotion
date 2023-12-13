defmodule PATHFINDER do   # mix run benchmarks/pathfinder.ex 100 100 10    nvcc c_src/Elixir.PATHFINDER_gp.cu
  import GPotion
  gpotion pathfinder(block_size, prev, tempResult, iteration, gpu_wall, gpu_src, gpu_results, cols, start_step, border) do
    # Implement the equivalent of the CUDA dynproc_kernel function using Matrex for GPU operations.

    bx = blockIdx.x
    tx = threadIdx.x

    validXmin = 0
    validXmax = 0
    isValid = 0
    computed = 0

    # each block finally computes result for a small block
    # after N iterations.
    # it is the non-overlapping small blocks that cover
    # all the input data

    # calculate the small block size
    small_block_cols = block_size - iteration * 2                 # - border*2

    # calculate the boundary for the block according to
    # the boundary of its small block
    blkX = small_block_cols * bx - border
    blkXmax = blkX + block_size - 1

    # calculate the global thread coordination
    xidx = blkX + tx

    # effective range within this block that falls within
    # the valid range of the input data
    # used to rule out computation outside the boundary.

    if (blkX < 0) do
      validXmin = -blkX
    end

    if (blkXmax > cols-1) do
      validXmax = block_size - (blkXmax - cols + 1) - 1
    else
      validXmax = block_size - 1
    end

    w = tx - 1
    e = tx + 1

    if (w < validXmin) do
      w = validXmin
    end

    if (e > validXmax) do
      e = validXmax
    end

    if (((tx-validXmin)*(validXmax-tx)) >= 0) do                  # equivalent to IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
      isValid = 1
    end

    if (((xidx-0)*(cols-1-xidx)) >= 0) do
      prev[tx] = gpu_src[xidx]
    end

    for i in range(0, iteration, 1) do
      computed = 0
      if ((((tx-i+1)*(block_size-i-2-tx)) >= 0) && isValid) do
        computed = 1
        left = prev[w]
        up = prev[tx]
        right = prev[e]
        shortest = min(left, up)
        shortest = min(shortest, right)
        index = cols*(start_step+i)+xidx
        tempResult[tx] = shortest + gpu_wall[index]
      end

      if (i == iteration - 1) do
        break                                                     # break the loop, we don't need to compute the rest of the iterations
      end

      if (computed) do
        # assign the computation range
        prev[tx] = tempResult[tx]
      end
    end

    # update the global memory
    # after the last iteration, only threads coordinated within the
    # small block perform the calculation and switch on ``computed''
    if (computed) do
      gpu_results[xidx] = tempResult[tx]                          # update the global memory, from temporary memory to global memory
    end
  end
end

#/* --------------- pyramid parameters --------------- */
block_size = 256                                                  # Define your BLOCK_SIZE value

[a1,a2,a3] = System.argv()

rows = String.to_integer(a1)                                     # get the matrix size from the command line
cols = String.to_integer(a2)
pyramid_height = String.to_integer(a3)

size = rows*cols
border_cols = pyramid_height                                      # number of elements in the border = pyramid_height * HALO = 7 * 1 = 7
small_block_cols = block_size - pyramid_height * 2                # number of columns in a single CUDA block = BLOCK_SIZE - (border*2) = 256 - (pyramid_height * 2)
block_cols = div(cols, small_block_cols) + if rem(cols, small_block_cols) == 0, do: 0, else: 1                                  # number of blocks in a single row

IO.puts "pyramidHeight: #{pyramid_height}"
IO.puts "gridSize:      #{cols}"
IO.puts "border:        #{border_cols}"
IO.puts "blockSize:     #{block_size}"
IO.puts "blockGrid:     #{block_cols}"
IO.puts "targetBlock:   #{small_block_cols}"

# Perform kernel launches

prevTime = System.monotonic_time()                                # get the current time
ker=GPotion.load(&PATHFINDER.pathfinder/10)

f = fn _ -> Enum.random(1..10) end
data = Matrex.fill(1,size,1)                                      # create a matrix of one row and (rows*cols) column's filled with 1
data = Matrex.apply(data,f)
wall = Matrex.submatrix(data, 1 .. 1, cols .. size)               # the wall begin at row 1, column (cols+1) of the input data

gpu_wall = GPotion.new_gmatrex(wall)

result = Matrex.zeros(1, cols)
gResult1 = GPotion.new_gmatrex(data)                              # the gpu_result[0] contain the input data
gResult2 = GPotion.new_gmatrex(result)

gpu_result = [gResult1, gResult2]

prevN = Matrex.zeros(1, block_size)                               #  prev[block_size]
prev = GPotion.new_gmatrex(prevN)

tempResultN = Matrex.zeros(1, block_size)                         #  result[block_size]
tempResult = GPotion.new_gmatrex(tempResultN)

src = 1                                                           #  source and destination tiling indices used for switching between ping-pong buffers
dst = 0

for t <- 0 .. rows-1 // pyramid_height do
  temp = src
  src = dst
  dst = temp
  iteration = min(pyramid_height, rows-t-1)
  GPotion.spawn(ker,{block_cols,1,1},{block_size,1,1},[block_size, prev, tempResult, iteration, gpu_wall, Enum.at(gpu_result, src), Enum.at(gpu_result, dst), cols, t, border_cols])
  GPotion.synchronize()
end

GPotion.synchronize()

finalResult = GPotion.get_gmatrex(Enum.at(gpu_result, dst))

IO.inspect(data)
IO.inspect(finalResult)

nextTime = System.monotonic_time()
IO.puts "GPotion Pyramid_Height:\t#{pyramid_height} Time:\t#{System.convert_time_unit(nextTime-prevTime,:native,:millisecond)}"
