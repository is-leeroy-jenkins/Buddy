from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.10.0+cpu'
debug = False
cuda: Optional[str] = None
git_version = '449b1768410104d3ed79d3bcfe4ba1d65c7f22c0'
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
