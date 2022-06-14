import uafgi.indexing
import numpy as np

def get_indexing(ndarr):
    """Assumes ndarr is fully packed, row-major: j,i indexing"""
    base = (0,0)
    extent = ndarr.shape
    indices = (0,1)    # List highest-stride index first
    return uafgi.indexing.Indexing(base, extent, indices)



def test_indexing():
    arr = np.array([
        [1,2,3],
        [4,5,6]])
    arr1 = np.reshape(arr, -1)
    ind = get_indexing(arr)
    indT = ind.transpose()

    for jj in range(0, arr.shape[0]):
        for ii in range(0, arr.shape[1]):
            print(jj,ii)
            k = ind.tuple_to_index((jj,ii))
            kT = indT.tuple_to_index((ii,jj))
            assert k == kT
            assert arr[jj,ii] == arr1[k]
