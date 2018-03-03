import os
import torch
from torch.utils.ffi import create_extension

if not torch.cuda.is_available():
    raise ValueError('cuda is not available')

this_file = os.path.dirname(__file__)

sources = ['dot_power/src/dot_power_cuda.cc']
headers = ['dot_power/src/dot_power_cuda.h']
defines = []

ffi = create_extension(
    'dot_power._ext.dot_power',
    package = True,
    headers = headers,
    sources = sources,
    define_macros = defines,
    relative_to=__file__,
    include_dirs=[ os.environ['CUDA_PATH'] + '/include',os.environ['DOT_POWER_PATH']+'/include'],
    libraries = ['dot_power','cudart','ATen','shm'],
    library_dirs = [os.environ['DOT_POWER_PATH']+'/lib',os.environ['DOT_POWER_PATH']+'/bin',os.environ['CUDA_PATH']+'/lib/x64'],
    with_cuda = True,
)

if __name__ == '__main__':
    ffi.build();
