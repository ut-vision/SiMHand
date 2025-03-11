N_VALID_KEYPOINTS = 10 # out of 21
VALID_SEQ_SAMPLE_RATE = 0.5 # seq batch should contain at least 50% valid samples
# mono keypoint order
WRIST_ID = 0
MIDDLE_MCP_ID = CENTER_ID = 9
REF_BONE_LINK = (WRIST_ID, MIDDLE_MCP_ID)
DEFAULT_CACHE_DIR = '.cache'
SNAP_JOINT_NAMES = [
    'loc_bn_palm_L',
    'loc_bn_thumb_L_01',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_04',
    'loc_bn_index_L_01',
    'loc_bn_index_L_02',
    'loc_bn_index_L_03',
    'loc_bn_index_L_04',
    'loc_bn_mid_L_01',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_04',
    'loc_bn_ring_L_01',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_04',
    'loc_bn_pinky_L_01',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_04'
]

RHD_JOINTS = [
    'loc_bn_palm_L',
    'loc_bn_thumb_L_04',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_01',
    'loc_bn_index_L_04',
    'loc_bn_index_L_03',
    'loc_bn_index_L_02',
    'loc_bn_index_L_01',
    'loc_bn_mid_L_04',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_01',
    'loc_bn_ring_L_04',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_01',
    'loc_bn_pinky_L_04',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_01'
]

STB_JOINTS = [
    'loc_bn_palm_L',
    'loc_bn_pinky_L_01',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_04',
    'loc_bn_ring_L_01',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_04',
    'loc_bn_mid_L_01',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_04',
    'loc_bn_index_L_01',
    'loc_bn_index_L_02',
    'loc_bn_index_L_03',
    'loc_bn_index_L_04',
    'loc_bn_thumb_L_01',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_04',
]

SNAP_BONES = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20)
]

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]

USEFUL_BONE = [1, 2, 3,
            5, 6, 7,
            9, 10, 11,
            13, 14, 15,
            17, 18, 19]

KINEMATIC_TREE = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

ID2ROT = {
        2: 13, 3: 14, 4: 15,
        6: 1, 7: 2, 8: 3,
        10: 4, 11: 5, 12: 6,
        14: 10, 15: 11, 16: 12,
        18: 7, 19: 8, 20: 9,
    }

JOINT_COLORS = (
    (216, 31, 53),
    (214, 208, 0),
    (136, 72, 152),
    (126, 199, 216),
    (0, 0, 230),
)

# from interhand to mano
JMAP_INTERHAND_TO_PANOPTIC =  [
                        20,             # wrist
                        3, 2, 1, 0,     # thumb                        
                        7, 6, 5, 4,      # index
                        11, 10, 9, 8,    # middle                        
                        15, 14, 13, 12,  # ring  
                        19, 18, 17, 16,  # pinky                                                                  
                    ]

# from mano to interhand
JMAP_PANOPTIC_TO_INTERHAND = [    
    4, 3, 2, 1, # thumb
    8, 7, 6, 5, # index
    12, 11, 10, 9, # middle
    16, 15, 14, 13, # ring
    20, 19, 18, 17, # pinky
    0, 
]


VAL_TEST_DATA_LIST = ['stb', 'rhd', 'do', 'eo', 'ah', 'dy']
TIP_ONLY_DATA_LIST = ['do', 'eo']
SEQ_DATA_LIST = ['dy', 'ah']

import os
MANO_MODEL_PATH = "minimal-hand/mano/models"
assert os.path.exists(MANO_MODEL_PATH), f"mano path not found: {MANO_MODEL_PATH}"

'''
from datasets.dexter_object import DexterObjectDataset
from datasets.egodexter import EgoDexter
from datasets.ganerated_hands import GANeratedDataset
from datasets.hand143_panopticdb import Hand143_panopticdb
from datasets.hand_labels import Hand_labels
from datasets.rhd import RHDDataset
from datasets.stb import STBDataset
from datasets.assembly_hands import AssemblyHandsDataset
from datasets.dexycb import DexYCBDataset
from datasets.handataset import HandDataset
from datasets.seqhandataset import SeqHandDataset
'''

def get_frame_dataset(dataset_name, split, args, **kwargs):
    if split == 'train':
        dataset = HandDataset(
                    data_split=split,
                    train=(split == 'train'),
                    subset_name=dataset_name,
                    data_root=args.data_root,
                    hand_side="right",
                    is_debug=args.debug,
                    **kwargs,
                )
    else:
        if dataset_name == 'eo':
            dataset = EgoDexter(
                        data_split=split,
                        data_root=args.data_root,
                        hand_side="right",
                    )
        elif dataset_name in VAL_TEST_DATA_LIST:
            dataset = HandDataset(
                    data_split=split,
                    train=(split == 'train'),
                    subset_name=[dataset_name],
                    data_root=args.data_root,
                    hand_side="right",
                    is_debug=args.debug,
                    **kwargs,
                )
        else:
            raise NotImplementedError("Unknown dataset: {}".format(dataset_name))
    
    return dataset

def get_seq_dataset(dataset_name, split, args, **kwargs):
    if isinstance(dataset_name, list):
        for subset_name in dataset_name:
            assert subset_name in SEQ_DATA_LIST
    else:
        assert dataset_name in SEQ_DATA_LIST
        dataset_name = [dataset_name]
    
    dataset = SeqHandDataset(
                data_split=split,
                train=(split == 'train'),
                subset_name=dataset_name,
                data_root=args.data_root,
                hand_side="right",
                is_debug=args.debug,
                seqlen=args.seqlen,
                stride=args.stride,
                no_img_load=args.no_img_load,
                **kwargs,
            )
    return dataset    
