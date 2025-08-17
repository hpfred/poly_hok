require PolyHok
use Ske

#n = 1000
n = 20

#arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})
arr1 = Nx.tensor(Enum.to_list(1..n),type: {:s, 32})
#arr2 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})
#arr2 = Nx.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

par1 = 0#100
par2 = 0#1000
par3 = 0#10000
par4 = PolyHok.new_gnx(arr1)

PolyHok.defmodule Teste do
    defd sum2(x,y,z,q,w,r) do
        x = x + y + z + q + w + r
    end
end

arr2 = PolyHok.new_gnx(arr2)
#IO.inspect(arr1)
arr1 =  arr1 |> PolyHok.new_gnx 

    ## MAP
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x) -> x + 1 end))
    ## MAP2
        #arr2 = arr2|> Ske.map2(arr2,PolyHok.phok(fn (x,y) -> x + y end))
        

    ## map_1_para_1D
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1],[coord: false, return: false, dim: :one])
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y) -> z = 1 end),[par4],return: false)
    
    ## map_1para_1D_resp
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1])
    
    ## map_1_para_coord_1D
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1],[coord: true, return: false, dim: :one])
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z) -> q = 1 end),[par1],[coord: true, return: false, dim: :one])
    
    ## map_1_para_coord_1D_resp
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1],[coord: true, return: true, dim: :one])
    
    ## map_1para_2D
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1],[coord: false, return: false, dim: :two])
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y) -> z = 1 end),[par1],[coord: false, return: false, dim: :two])
    
    ## map_1para_2D_resp
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1],[coord: false, return: true, dim: :two])
    
    ## map_1para_coord_2D
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1],[coord: true, return: false, dim: :two])
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> r = 1 end),[par1],[coord: true, return: false, dim: :two])
    
    ## map_1para_coord_2D_resp
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1],[coord: true, return: true, dim: :two])
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> z end),[par1],[coord: true, return: true, dim: :two])
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + q end),[par1],[coord: true, return: true, dim: :two])
    
    ## map_2para_1D
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: false, dim: :one])
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z) -> q = 1 end),[par1,par2],[coord: false, return: false, dim: :one])
    
    ## map_2para_1D_resp
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: true, dim: :one])
    
    ## map_2para_coord_1D
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2],[coord: true, return: false, dim: :one])
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> r = 1 end),[par1,par2],[coord: true, return: false, dim: :one])
    
    ## map_2para_coord_1D_resp
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2],[coord: true, return: true, dim: :one])
    
    ## map_2para_2D
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: false, dim: :two])
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z) -> q = 1 end),[par1,par2],[coord: false, return: false, dim: :two])
    
    ## map_2para_2D_resp
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: true, dim: :two])
    
    ## map_2para_coord_2D
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2],[coord: true, return: false, dim: :two])
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> r = 1 end),[par1,par2],[coord: true, return: false, dim: :two])
    
    ## map_2para_coord_2D_resp
        #arr2 = arr2|> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2],[coord: true, return: true, dim: :two])
    
    ## map_3para_1D
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: false, dim: :one])
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> r = 1 end),[par1,par2,par3],[coord: false, return: false, dim: :one])
    
    ## map_3para_1D_resp
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: true, dim: :one])
    
    ## map_3para_coord_1D
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2,par3],[coord: true, return: false, dim: :one])
        #arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> r = 1 end),[par1,par2,par3],[coord: true, return: false, dim: :one])
    
    ## map_3para_coord_1D_resp
        #arr1 = arr1 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2,par3],[coord: true, return: true, dim: :one])
    
    ## map_3para_2D
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: false, dim: :two])
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> r = 1 end),[par1,par2,par3],[coord: false, return: false, dim: :two])
    
    ## map_3para_2D_resp
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: true, dim: :two])
    
    ## map_3para_coord_2D
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> x + y + z + q + w + r end),[par1,par2,par3],[coord: true, return: false, dim: :two])
        #arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> s = 1 end),[par1,par2,par3],[coord: true, return: false, dim: :two])
        #arr2 |> Ske.map(&Teste.sum6/6,[par1,par2,par3],[coord: true, return: false, dim: :two])
    
    ## map_3para_coord_2D_resp
        arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> x + y + z + q + w + r end),[par1,par2,par3],[coord: true, return: true, dim: :two])
        
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> w + r end),[par1,par2,par3],[coord: true, return: true, dim: :two])
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> w end),[par1,par2,par3],[coord: true, return: true, dim: :two])
        #arr2 = arr2 |> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> r end),[par1,par2,par3],[coord: true, return: true, dim: :two])


#IO.inspect(arr1)
#PolyHok.get_gnx(arr1) |> IO.inspect
#PolyHok.get_gnx(arr2) |> IO.inspect
arr2 = PolyHok.get_gnx(arr2)
IO.inspect(arr2,limit: 4096)
