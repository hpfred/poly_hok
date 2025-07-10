require PolyHok
use Ske

#n = 1000
n = 20

arr1 = Nx.tensor(Enum.to_list(1..n),type: {:s, 32})
par1 = 1
par2 = 1
par3 = 1
par4 = PolyHok.new_gnx(arr1)
arr2 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})
arr2 = PolyHok.new_gnx(arr2)


arr1 =  arr1 |> PolyHok.new_gnx
    ## MAP
        #|> Ske.map(PolyHok.phok(fn (x) -> x + 1 end))
    ## MAP2
        #|> Ske.map2(arr2,PolyHok.phok(fn (x,y) -> x + y end))
        
    ## map_1_para_1D
        #|> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1],[coord: false, return: false, dim: :one])
        #|> Ske.map(PolyHok.phok(fn (x,y) -> y[0] = 1 end),[par4],return: false)
    ## map_1para_1D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1])
    ## map_1_para_coord_1D
        #|> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1],[coord: true, return: false, dim: :one])
    ## map_1_para_coord_1D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1],[coord: true, return: true, dim: :one])
    ## map_1para_2D
        #|> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1],[coord: false, return: false, dim: :two])
    ## map_1para_2D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y) -> x + y end),[par1],[coord: false, return: true, dim: :two])
    ## map_1para_coord_2D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1],[coord: true, return: false, dim: :two])
    ## map_1para_coord_2D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1],[coord: true, return: true, dim: :two])
    ## map_2para_1D
        #|> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: false, dim: :one])
    ## map_2para_1D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: true, dim: :one])
    ## map_2para_coord_1D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2],[coord: true, return: false, dim: :one])
    ## map_2para_coord_1D_resp
        |> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2],[coord: true, return: true, dim: :one])
    ## map_2para_2D
        #|> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: false, dim: :two])
    ## map_2para_2D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z) -> x + y + z end),[par1,par2],[coord: false, return: true, dim: :two])
    ## map_2para_coord_2D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2],[coord: true, return: false, dim: :two])
    ## map_2para_coord_2D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2],[coord: true, return: true, dim: :two])
    ## map_3para_1D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: false, dim: :one])
    ## map_3para_1D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: true, dim: :one])
    ## map_3para_coord_1D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2,par3],[coord: true, return: false, dim: :one])
    ## map_3para_coord_1D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w) -> x + y + z + q + w end),[par1,par2,par3],[coord: true, return: true, dim: :one])
    ## map_3para_2D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: false, dim: :two])
    ## map_3para_2D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q) -> x + y + z + q end),[par1,par2,par3],[coord: false, return: true, dim: :two])
    ## map_3para_coord_2D
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> x + y + z + q + w + r end),[par1,par2,par3],[coord: true, return: false, dim: :two])
    ## map_3para_coord_2D_resp
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> x + y + z + q + w + r end),[par1,par2,par3],[coord: true, return: true, dim: :two])
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> w + r end),[par1,par2,par3],[coord: true, return: true, dim: :two])
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> w end),[par1,par2,par3],[coord: true, return: true, dim: :two])
        #|> Ske.map(PolyHok.phok(fn (x,y,z,q,w,r) -> r end),[par1,par2,par3],[coord: true, return: true, dim: :two])
    
    
#IO.inspect(arr1)
PolyHok.get_gnx(arr1) |> IO.inspect
#IO.puts(arr1[999])
