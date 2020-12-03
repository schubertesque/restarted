import os
import numpy as np
from pyquaternion.quaternion import Quaternion as Q
import pickle
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import kinematics as K
from visualization import draw_obj
import parser 

# get skeleton. for testing, we just check against one body part.
def get_skeletons(body_part):
    for i_data, data_dict in enumerate(hoi_data):
    if i_data in [65, 67, 78, 97, 103, 131, 144, 149, 150, 151, 152, 153, 160, 173, 177, 178]:   # incorrect sitting
        continue
    if i_data in [14, 68, 69, 70, 71, 129, 130, 131, 136, 137, 138, 139, 140, 141, 169]:   # incorrect labeling -> incorrect relationship 130? 169?
        continue
    
    # extract obj bboxes
    obj = data_dict['object']
    if obj['hash'] not in bbox_dict:
        continue
    # extract skeleton
    skeleton = data_dict['person']['skeleton']
    
    # extract some other part
    if (body_part == "SKEL_PELVIS"):
        coords = skeleton['SKEL_PELVIS']    
        all_parts.append(coords)
        
    elif (body_part == "SKEL_L_THIGH"):
        coords = skeleton['SKEL_L_THIGH']
        all_parts.append(coords)
        
    elif (body_part == "SKEL_R_THIGH"):
        coords = skeleton['SKEL_R_THIGH']
        all_parts.append(coords)
        
    skeleton = np.array([skeleton[x] for x in K.joint_names]).astype(np.float32)
    center = np.array(data_dict['person']['root_position']).astype(np.float32)
    skeleton -= center
    
    rotation = np.array(data_dict['person']['root_rotation']).astype(np.float32)[[3,0,1,2]]
    skeleton = np.matmul(Q(rotation).inverse.rotation_matrix, skeleton.T).T
    
    obj_config = []
    obj_cat_id = []
    for part_id in bbox_dict[obj['hash']]:
        for bbox in bbox_dict[obj['hash']][part_id]:
            if bbox is None:
                continue
            T1,T2,T3,D,W,H,R = bbox
            R2 = Q(rotation).inverse * Q(np.array(obj['root_rotation'])[[3,0,1,2]])
            dx, dy, dz = (float(x) for x in obj['root_position'])
            dx -= center[0]
            dy -= center[1]
            dz -= center[2]
            obj_config.append([T3,T2,T1,D,W,H,*R.rotation_matrix[:,:2].reshape([6]),dx,dy,dz,*R2.rotation_matrix[:,:2].reshape([6])])
            obj_cat_id.append(part_id)

    all_skeletons.append(skeleton)
    all_obj_configs.append(obj_config)
    all_obj_cat.append(obj_cat_id)
    
def get_all_chairs():
    chair_folder = '/content/drive/My Drive/Colab Notebooks/GraphNN/OBB/Chair'
    for d in os.listdir(chair_folder):
        all_chair.append(d[:-4])

def match_with_annotations():  
    annotations_folder = '/content/drive/My Drive/Colab Notebooks/GraphNN/annotations'
    for d in os.listdir(annotations_folder):
    for x in os.listdir('/content/drive/My Drive/Colab Notebooks/GraphNN/annotations/{}'.format(d)):
        current_dir = '/content/drive/My Drive/Colab Notebooks/GraphNN/annotations/{}'.format(d)
        if (x == "meta.json"):
        current_file = '{}/meta.json'.format(current_dir)
        with open(current_file) as to_parse:
            json_output = json.load(to_parse)
            model = json_output['model_id']
            if (model in all_chair):
            models.append(model)
            folders.append(d)
            model_folder.append([model,d])
        
def load_jsons():
    for d in os.listdir(annotations_folder):
    if (d not in folders):
        continue
    else:
        for x in os.listdir('/content/drive/My Drive/Colab Notebooks/GraphNN/annotations/{}'.format(d)):
        current_dir = '/content/drive/My Drive/Colab Notebooks/GraphNN/annotations/{}'.format(d)
        if (x == "result.json"):
            current_file = '{}/result.json'.format(current_dir)
            with open(current_file) as to_parse:
            json_output = json.load(to_parse)
            json_to_parse.append(json_output)
            obj_id = model_folder.find(d)
            current_obj = obj_id[0]
            obj_ids.append(current_obj)
  
def parse_all_chairs():
    for i in range(len(json_to_parse)):
        parser.parse_chair(json_to_parse[i])
        output_parsed.append(parser)
        json_marker.append(obj_ids[i])
  
def match_leaf_nodes_with_data():        
    updated_parsed = []
    root_features = None
    for i in range(len(output_parsed)):
        dim_21 = []
        indices = []
        current_cat = all_obj_cat[i]
        current_skeleton = all_skeletons[i]
        current_config = all_obj_configs[i]
    
    # add leaf nodes 
        for i in range(len(current_config)):
            # if in destination
            if (current_cat[i] in output_parsed[i][1]):
            dim_21.append(current_cat[i])
            indices.append(output_parsed[i][1])
        
        # add root node feature
        root_features = current_skeleton
    
    # repeat this for each graph
    def send_to_graph(current_id):
            square_edges = pd.DataFrame(
            {"source": parsed_chair[0][0], "target": parsed_chair[0][1]}
        )

        square_node_data = pd.DataFrame(
            {"attributes": dim_21}, index=indices
        )
        
        frames = [square_edges, square_node_data]
        result = pd.concat(frames)

        result.to_pickle("./{}/pkl".format(obj_id_))

        square_edges = pd.DataFrame(
        {"source": parsed_chair[0][0], "target": parsed_chair[0][1]}
        )

        square_node_data = pd.DataFrame(
        {"attributes": dim_21}, index=indices
        )

        frames = [square_edges, square_node_data]
        result = pd.concat(frames)

        with open("./{}.pkl".format(obj_id), 'w') as f: 
        pickle.dump(frames, root_features)


def main():
    
    all_skeletons = []    # skeletons (full)
    all_obj_configs = []  # set of 21 dimensional arrays
    all_obj_cat = []      # names of leaf nodes corresponding to 21-dim arrays
    all_parts = []        # get just a single skeleton part (for testing)
    get_skeletons("SKEL_PELVIS") # change this
    
    all_chair = [] # all chairs in OBB folder
    get_all_chairs() # all chair IDs
        
    # match chair IDs with their correspoding annotations so that we can parse
    folders = []
    models = []
    model_folder = []
    match_with_annotations()

    json_to_parse = [] # result.json type files
    obj_ids = []  # object hash of chairs
    load_jsons() # load all json files corresponding to chairs
    
    output_parsed = [] # parser output. each element is a 2D array consisting of source and destination node pairs for that chair
    json_marker = [] # same as obj_ids
    parse_all_chairs() 
    
    # Create association between person and object.
    index = 0
    hashes = [] # chair
    hashx = [] # person
    association = {} # object: list of persons
    for sit in sit_data:
    person = sit['object']
    hash1 = person['hash']
    if (hash1 in all_chair):
        hashx.append(hash1)
        hashes.append(index)
        if hash1 not in association:
            association[hash1] = index
            
        else:
            association[hash1].append(index)
            
    index = index + 1
    
    match_leaf_nodes_with_data() # add features to leaf nodes
    
if __name__ == "__main__":
    main()