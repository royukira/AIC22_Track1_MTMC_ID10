from yacs.config import CfgNode as CN

_C = CN()

_C.MAIN_DIR = ''

_C.CHALLENGE_DATA_DIR = ''
_C.DET_SOURCE_DIR = ''
_C.REID_MODEL = ''
_C.REID_BACKBONE = ''
_C.REID_SIZE_TEST = [256, 256]

_C.DET_IMG_DIR = ''
_C.DATA_DIR = ''
_C.ROI_DIR = ''
_C.CID_BIAS_DIR = ''

_C.USE_RERANK = False
_C.USE_FF = False
_C.SCORE_THR = 0.5

_C.MCMT_OUTPUT_TXT = ''


# Add by Roy
_C.OUTPUT_DIR = ''
_C.CAM_SEQ = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']           # test/s06
_C.CAM_CLUSTER = [[41, 42, 43, 44, 45, 46]]
_C.ENSEMBLE_SEQ = ['detect_reid1', 'detect_reid2', 'detect_reid3']  
# _C.FEATUER_DIR = ''
