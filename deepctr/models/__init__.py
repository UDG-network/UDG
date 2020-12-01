from .afm import AFM
from .autoint import AutoInt
from .ccpm import CCPM
from .dcn import DCN
from .deepfm import DeepFM
from .deepfm_udg import DeepFM_UDG
from .pnn_udg import PNN_UDG
from .wdl_udg import WDL_UDG
from .dien import DIEN
from .dien_udg import DIEN_UDG
from .din import DIN
from .din_udg import DIN_UDG
from .fnn import FNN
from .mlr import MLR
from .onn import ONN
from .onn import ONN as NFFM
from .nfm import NFM
from .pnn import PNN
from .wdl import WDL
from .xdeepfm import xDeepFM
from .fgcnn import FGCNN
from .dsin import DSIN
from .fibinet import FiBiNET
from .flen import FLEN
from .fwfm import FwFM

__all__ = ["AFM", "CCPM","DCN", "MLR", "DeepFM", "MLR", "NFM", "DIN", "DIEN", "FNN", "PNN",
           "WDL", "xDeepFM", "AutoInt", "ONN", "FGCNN", "DSIN", "FiBiNET", 'FLEN', "FwFM"]
