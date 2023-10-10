# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import open3d as o3d
import numpy as np 
import json
import re
import math
import tqdm
from matplotlib import pyplot as plt
import plotly.graph_objects as go	
import plotly.offline as pyo
from kosmos_encode_box import classes_token_map

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

import numpy as np

# import pc_util

def cal_iou_3d(bbox1, bbox2):
    '''
        box [x1, y1, z1, l, w, h]
    '''
    bbox1 = [
        round(bbox1[0] - abs(bbox1[3]/2), 3), round(bbox1[1] - abs(bbox1[4]/2), 3), round(bbox1[2] - abs(bbox1[5]/2), 3), 
        round(bbox1[0] + abs(bbox1[3]/2), 3), round(bbox1[1] + abs(bbox1[4]/2), 3), round(bbox1[2] + abs(bbox1[5])/2, 3)
        ]
    
    bbox2 = [
        round(bbox2[0]-abs(bbox2[3]/2),3), round(bbox2[1]-abs(bbox2[4]/2),3), round(bbox2[2]-abs(bbox2[5]/2),3), 
        round(bbox2[0]+abs(bbox2[3]/2),3), round(bbox2[1]+abs(bbox2[4]/2),3), round(bbox2[2]+abs(bbox2[5])/2,3)
        ]
    # print('box_1', bbox1)
    # print('box_2', bbox2)
    # intersection
    x1, y1, z1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2])
    x2, y2, z2 = min(bbox1[3], bbox2[3]), min(bbox1[4], bbox2[4]), min(bbox1[5], bbox2[5])
    inter_area = max((x2 - x1), 0) * max((y2 - y1), 0) * max((z2 - z1), 0)
    # print('inter',inter_area)
    # union
    area1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2])
    area2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])
    uni_area = area1 + area2 - inter_area
    
    iou = inter_area / uni_area
    # print(inter_area, uni_area)
    
    if iou > 1 or iou < 0:
        return 0
    else:
        return iou
    
def get_box_coords_from_index_3d(P, ul_idx, lr_idx):  
    """  
    Given a grid of length P and the indices of the upper-left and lower-right corners of a bounding box,  
    returns the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2].  
      
    Args:  
    - P (int): the length of the grid  
    - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box  
    - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box  
      
    Returns:  
    - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]  
    """  
    # Compute the size of each cell in the grid  
    cell_size = 1.0 / P  
      
    # Compute the x and y indices of the upper-left and lower-right corners of the bounding box  
    code_z = ul_idx // P**2
    reming = ul_idx - code_z * P**2
    code_x = reming % P  
    code_y = reming // P
    

    # print(code_x, code_y, code_z) 
      
    code_w = lr_idx // P**2
    reming = lr_idx - code_w * P**2
    code_c = reming % P  
    code_h = reming // P
    # print(code_c, code_h, code_w)
    # print(code_x, code_y, code_z, code_c, code_h, code_w) 
      
    # Compute the normalized coordinates of the bounding box  
    # if ul_idx == lr_idx:  
    #     x = ul_x * cell_size  
    #     y = ul_y * cell_size  
    #     c = lr_x * cell_size + cell_size  
    #     h = lr_y * cell_size + cell_size  
    # elif ul_x == lr_x or ul_y == lr_y:  
    #     x1 = ul_x * cell_size  
    #     y1 = ul_y * cell_size  
    #     x2 = lr_x * cell_size + cell_size  
    #     y2 = lr_y * cell_size + cell_size  
    # else:  
    x = code_x * cell_size + cell_size / 2  
    y = code_y * cell_size + cell_size / 2  
    z = code_z * cell_size + cell_size / 2 

    c = code_c * cell_size + cell_size / 2  
    h = code_h * cell_size + cell_size / 2  
    w = code_w * cell_size + cell_size / 2  
      
    return np.array([x, y, z, c, h, w])

def get_box_coords_from_location_3d(P, box_list):
    x, y, z, c, h, w = box_list[0], box_list[1], box_list[2], box_list[3], box_list[4], box_list[5]
    return np.array([float(x/P), float(y/P), float(z/P), float(c/P), float(h/P), float(w/P)])

def parse_bbox_3d_decode_index(text):
    num_list = []
    # 首先是从text中抽取出所有的数字信息
    # num_list = parse_num(num_list, '[', ']', text)
    # num_list = parse_num(num_list, '(', ')', text)
    pattern = r'<patch_index_\d+> <patch_index_\d+>'  
    # 从输出结果中找到相对应的输出模板
    # Find all matches in the given string  
    matches = re.findall(pattern, text)
    
    for match in matches:
        patch_index_pairs = match.split(' ')
        # print(patch_index_pairs)
        
        patch_index_xyz = re.search(r'<patch_index_(\d+)>', patch_index_pairs[0])  
        patch_index_chw = re.search(r'<patch_index_(\d+)>', patch_index_pairs[1])  
        num_list.append((int(patch_index_xyz.group(1)), int(patch_index_chw.group(1))))

    bbox_list = []
    for box_index in num_list:
        resume_box = get_box_coords_from_index_3d(16, box_index[0], box_index[1])
        unnormalized_bbox = [-math.log((1 / (resume_box[i] + 1e-8)) - 1) for i in range(0, len(resume_box))]
        # 对输出的box坐标进行反向归一化
        bbox_list.append(list(unnormalized_bbox))
    
    return bbox_list

def parse_bbox_3d_decode_location(text):
    num_list = []
    class_token_list = []
    # 首先是从text中抽取出所有的数字信息
    # num_list = parse_num(num_list, '[', ']', text)
    # num_list = parse_num(num_list, '(', ')', text)
    pattern = r'<patch_index_\d+> <patch_index_\d+> <patch_index_\d+> <patch_index_\d+> <patch_index_\d+> <patch_index_\d+> <c\d+>'  
    # 从输出结果中找到相对应的输出模板
    # Find all matches in the given string  
    matches = re.findall(pattern, text)
    # print(len(matches))
    # print(matches)
    for match in matches:
        patch_index_pairs = match.split(' ')
        # print(match)
        x = re.search(r'<patch_index_(\d+)>', patch_index_pairs[0])  
        y = re.search(r'<patch_index_(\d+)>', patch_index_pairs[1]) 
        z = re.search(r'<patch_index_(\d+)>', patch_index_pairs[2]) 
        c = re.search(r'<patch_index_(\d+)>', patch_index_pairs[3]) 
        h = re.search(r'<patch_index_(\d+)>', patch_index_pairs[4]) 
        w = re.search(r'<patch_index_(\d+)>', patch_index_pairs[5]) 
        num_list.append((int(x.group(1)), int(y.group(1)), int(z.group(1)), int(c.group(1)), int(h.group(1)), int(w.group(1))))
        class_token_list.append(patch_index_pairs[6])
        
    bbox_list = []
    class_list = []
    # print(num_list)
    for box_index in num_list:
        resume_box = get_box_coords_from_location_3d(1000, box_index)
        # print(resume_box)
        unnormalized_bbox = [-math.log((1 / (resume_box[i] + 1e-8)) - 1) for i in range(0, len(resume_box))]
        # 对输出的box坐标进行反向归一化
        bbox_list.append(list(unnormalized_bbox))
    
    for class_token in class_token_list:
        for token_name in classes_token_map.keys():
            # print(classes_token_map[token_name], class_token)
            if classes_token_map[token_name][0] == class_token:
                class_list.append(classes_token_map[token_name][0])
    
    # print(class_list)
            
    return bbox_list

def compute_box_3d(center, size, heading_angle=0,color=[0,1,0]):

    h = size[2]#z
    w = size[1]#y
    l = size[0]#x
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])

    #heading_angle = -heading_angle - np.pi / 2 
    # center[2] = center[2] + h / 2
    R = rotz(1*heading_angle)
    l = l/2
    w = w/2
    h = h/2
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    corners_3d=np.transpose(corners_3d)
    bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [color for _ in range(len(bbox_lines))] # red
    bbox = o3d.geometry.LineSet()
    bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
    bbox.colors = o3d.utility.Vector3dVector(colors)
    bbox.points = o3d.utility.Vector3dVector(corners_3d)

    return bbox

def dict2json(file_name,the_dict):
    '''
    将字典文件写如到json文件中
    :param file_name: 要写入的json文件名(需要有.json后缀),str类型
    :param the_dict: 要写入的数据,dict类型
    :return: 1代表写入成功,0代表写入失败
    '''
    try:
        json_str = json.dumps(the_dict, indent=4)
        with open(file_name, 'w') as json_file:
            json_file.write(json_str)
        return 1
    except:
        return 0

def visualize_detection_box():
    scene_name = 'scannet_train_detection_data/scene0005_00'
    output_folder = 'data_viz_dump'

    data = np.load(scene_name+'_vert.npy')
    scene_points = data[:,0:3]
    colors = data[:,3:]
    instance_labels = np.load(scene_name+'_ins_label.npy')
    semantic_labels = np.load(scene_name+'_sem_label.npy')
    instance_bboxes = np.load(scene_name+'_bbox.npy')

    print(np.unique(instance_labels))
    print(np.unique(semantic_labels))
    # input()
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    b = np.array([1 , 1, 1,255, 255, 255]) # 每一列要除的数
    np.savetxt('./scene.txt', data[:,:6]/b)

    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="{}".format('detection_res'))
    #设置点云大小
    vis.get_render_option().point_size = 3
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    #创建点云对象
    pcd =o3d.io.read_point_cloud('./scene.txt', format='xyzrgb')
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)


    for box in instance_bboxes:
        print(box)
        box = compute_box_3d(box[:3], box[3:6])
        vis.add_geometry(box) 

    vis.run()
    vis.destroy_window()
    # from model_util_scannet import ScannetDatasetConfig
    # DC = ScannetDatasetConfig()
    # print(instance_bboxes.shape)

def visualize_scanrefer_benchmark_box():
    VG_path = '/media/xhm/Elements/ScanNet_Vis/ScanRefer_v5/1400/vicuna_13b_v0_4gpu_use_enc_ft_v_detr_hiddenlayer_-3/VG_ScanRefer_2000.json'
    VG_GT_path = './VG_ScanRefer_600_GT.json'
    
    # np.set_printoptions(suppress=True) # 取消默认科学计数法，open3d无法读取科学计数法表示
    # data = np.load('/home/xhm/Desktop/Code/3D/V-DETR_LLM/data/3D-Benchmark/scannet_pcls/scene0011_00_vert.npy')
    # b = np.array([1 , 1, 1,255, 255, 255]) # 每一列要除的数
    # np.savetxt('./scene.txt', data[:,:6]/b)
    # # 读取点云并可视化
    # pcd =o3d.io.read_point_cloud('./scene.txt', format='xyzrgb') # 原npy文件中的数据正好是按x y z r g b进行排列
    # print(pcd)
    # o3d.visualization.draw_geometries([pcd], width=1200, height=600)
    with open(VG_GT_path) as f:
        GT = json.load(f)
    
    with open(VG_path) as f:
        predict = json.load(f)
    
    number = 45 # 3 40 25
    test_scane = GT[number]
    predict_scene = predict[number]
    bboxes = parse_bbox_3d_decode_location(predict_scene['text'])
    predict_box = bboxes[0] + [0.0]
    print(predict_box)
    # assert 1 == 2
    
    id = test_scane['id']
    question = test_scane['question']
    print(id)
    
    raw_point = np.load('/home/xhm/Desktop/Code/3D/V-DETR_LLM/data/3D-Benchmark/scannet_pcls/{}_vert.npy'.format(id)) #读取1.npy数据  N*[x,y,z]
    b = np.array([1 , 1, 1 ,255 , 255, 255]) # 每一列要除的数
    np.savetxt('./scene.txt', raw_point[:,:6]/b)

    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="{}".format(question))
    #设置点云大小
    vis.get_render_option().point_size = 0
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    #创建点云对象
    pcd =o3d.io.read_point_cloud('./scene.txt', format='xyzrgb')
    #将点云数据转换为Open3d可以直接使用的数据类型
    # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)
    
    
    box = compute_box_3d(test_scane['object'][:3], test_scane['object'][3:6], color=[0,1,0])
    vis.add_geometry(box) 
    
    pred = compute_box_3d(predict_box[:3], predict_box[3:6], color=[1,0,0])
    vis.add_geometry(pred) 

    vis.run()
    vis.destroy_window()

def visualize_scannet_benchmark_box():
    Detec_path = './Scannet_Detection_Instruction_v4_normalized/1400/10_2560_30/vicuna_13b_v0_4gpu_use_enc_ft_v_detr_hiddenlayer_-3_six_token_embedding/Detection_ScanNet_v3.json'
    Detec_GT_path = './Detection_ScanNet_v3.json'
    
    with open(Detec_GT_path) as f:
        GT = json.load(f)
    
    with open(Detec_path) as f:
        predict = json.load(f)

    print('GT:',len(GT))
    print('prediction:',len(predict))
    
    # assert 1==2
    
    number = 40 # 40 20  86 pianyi  96 108 120 # training 3 5 8 10 12 14 18 24
    test_scane = GT[number]
    predict_scene = predict[number]
    bboxes = parse_bbox_3d_decode_location(predict_scene['text']) # 从predict中提取出需要的box
    
    print('length of predict bboxes:', len(bboxes))
    
    print('length of ground truth:', len(test_scane['object']))
    
    id = test_scane['id']
    # /home/xhm/Desktop/Code/3D/V-DETR_LLM/data/3D-Benchmark/scannet_pcls  ./scannet_train_detection_data
    raw_point = np.load('./scannet_train_detection_data/{}.npy'.format(id)) #读取1.npy数据  N*[x,y,z]
    b = np.array([1 , 1, 1,255, 255, 255]) # 每一列要除的数
    np.savetxt('./scene.txt', raw_point[:,:6]/b)

    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="{}".format('detection_res'))
    #设置点云大小
    vis.get_render_option().point_size = 0
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    #创建点云对象
    pcd =o3d.io.read_point_cloud('./scene.txt', format='xyzrgb')
    #将点云数据转换为Open3d可以直接使用的数据类型
    # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)
    
    
    print("GT:",test_scane['object'][0]['bbox']+[0.0])
    for gt_box in test_scane['object']:
        # print(gt_box['label'])
        # bbox = [-math.log((1 / (gt_box['bbox'][i] + 1e-8)) - 1) for i in range(0, len(gt_box['bbox']))]
        bbox = gt_box['bbox']
        box = compute_box_3d([(bbox[0]+bbox[3])/2, (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2], [(bbox[3]-bbox[0]), (bbox[4]-bbox[1]), (bbox[5]-bbox[2])])
        vis.add_geometry(box) 
    
    best_iou_box = []
    for pred_box in bboxes:
        predict_iou = []
        for gt_box in test_scane['object']:
            # bbox = [-math.log((1 / (gt_box['bbox'][i] + 1e-8)) - 1) for i in range(0, len(gt_box['bbox']))]
            # gt_bbox = [(bbox[0]+bbox[3])/2, (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2, 
            #            (bbox[3]-bbox[0]), (bbox[4]-bbox[1]), (bbox[5]-bbox[2])]
            
            gt_bbox = [(gt_box['bbox'][0]+gt_box['bbox'][3])/2, (gt_box['bbox'][1]+gt_box['bbox'][4])/2, (gt_box['bbox'][2]+gt_box['bbox'][5])/2, 
                       (gt_box['bbox'][3]-gt_box['bbox'][0]),   (gt_box['bbox'][4]-gt_box['bbox'][1]),   (gt_box['bbox'][5]-gt_box['bbox'][2])]
            
            # pred_box = [(pred_box[0]+pred_box[3])/2, (pred_box[1]+pred_box[4])/2, (pred_box[2]+pred_box[5])/2, 
            #             (pred_box[3]-pred_box[0]), (pred_box[4]-pred_box[1]), (pred_box[5]-pred_box[2])]
            
            if cal_iou_3d(gt_bbox, [pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_box[5], pred_box[4]]) >= 0.1:
                # print("GT",gt_bbox)
                # print("Pred", pred_box)
                print('IoU',cal_iou_3d(gt_bbox, [pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_box[5], pred_box[4]])) # 这里的顺序是发生了变化的所以对应的训练顺序也是有问题
                best_iou_box.append(pred_box)
    
    for best_box in best_iou_box:
        box = compute_box_3d(best_box[:3], [best_box[3], best_box[5], best_box[4]], color=[1,0,0])
        vis.add_geometry(box) 
    

    vis.run()
    vis.destroy_window()

def visualize_scannet_finetune_box():
    Detec_GT_path = './Scannet_Detection_Instruction_v4.json'
    
    with open(Detec_GT_path) as f:
        GT = json.load(f)
        
    number = 50 # 40 
    test_scane = GT[number]
    
    id = test_scane['pcl'].split("/")[2]
    # /home/xhm/Desktop/Code/3D/V-DETR_LLM/data/3D-Benchmark/scannet-pcls  ./scannet_train_detection_data
    raw_point = np.load('./scannet_train_detection_data/{}'.format(id)) #读取1.npy数据  N*[x,y,z]
    b = np.array([1 , 1, 1, 255, 255, 255]) # 每一列要除的数
    np.savetxt('./scene.txt', raw_point[:,:6]/b)

    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="{}".format('scannet_detection'))
    #设置点云大小
    vis.get_render_option().point_size = 0
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    #创建点云对象
    pcd =o3d.io.read_point_cloud('./scene.txt', format='xyzrgb')
    #将点云数据转换为Open3d可以直接使用的数据类型
    # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    #设置点的颜色为白色
    pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)
    
    
    # print("GT:",test_scane['conversations'][1]['value'])
    for gt_box in test_scane['conversations'][1]['value']:
        # print(gt_box['label'])
        bbox = gt_box[1:]
        # bbox = [(gt_box[0]+gt_box[3])/2, (gt_box[1]+gt_box[4])/2, (gt_box[2]+gt_box[5])/2, 
        #         (gt_box[3]-gt_box[0]), (gt_box[4]-gt_box[1]), (gt_box[5]-gt_box[2])]
        # print(bbox)
        # print(bbox[:3], bbox[3:])
        box = compute_box_3d(bbox[:3], bbox[3:])
        vis.add_geometry(box) 
    

    vis.run()
    vis.destroy_window()
    
def visualize_scanrefer_finetune_box():
    Detec_GT_path = './ScanRefer_objname.json'
    
    with open(Detec_GT_path) as f:
        GT = json.load(f)
        
    number = 256 # 40 
    test_scane = GT[number]
    
    id = test_scane['pcl'].split("/")[2]
    question = test_scane['conversations'][0]['value']
    # /home/xhm/Desktop/Code/3D/V-DETR_LLM/data/3D-Benchmark/scannet-pcls  ./scannet_train_detection_data
    raw_point = np.load('./scannet_train_detection_data/{}'.format(id)) #读取1.npy数据  N*[x,y,z]
    b = np.array([1 , 1, 1,255, 255, 255]) # 每一列要除的数
    np.savetxt('./scene.txt', raw_point[:,:6]/b)

    #创建窗口对象
    vis = o3d.visualization.Visualizer()
    #设置窗口标题
    vis.create_window(window_name="{}".format(question))
    #设置点云大小
    vis.get_render_option().point_size = 3
    #设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255, 255, 255])

    #创建点云对象
    pcd =o3d.io.read_point_cloud('./scene.txt', format='xyzrgb')
    #将点云数据转换为Open3d可以直接使用的数据类型
    # pcd.points= o3d.open3d.utility.Vector3dVector(raw_point)
    #设置点的颜色为白色
    # pcd.paint_uniform_color([1,1,1])
    #将点云加入到窗口中
    vis.add_geometry(pcd)
    
    
    # print("GT:",test_scane['conversations'][1]['value'])
    bbox = test_scane['conversations'][1]['value']
        # print(gt_box['label'])
        # bbox = gt_box[1:]
    print(bbox)
    print(bbox[:3], bbox[3:])
    box = compute_box_3d(bbox[:3], bbox[3:])
    vis.add_geometry(box) 
    

    vis.run()
    vis.destroy_window()
    
def creat_scannet_json():
    
    TRAIN_SCAN_NAMES = [line.rstrip() for line in open('/media/xhm/Elements/votenet/scannet/meta_data/scannetv2_train.txt')]
    
    single_room = {'cabinet':[], 'bed':[], 'chair':[], 'sofa':[], 'table':[], 'door':[], 'window':[],
             'bookshelf':[], 'picture':[], 'counter':[], 'desk':[], 'curtain':[],
             'refrigerator':[], 'showercurtrain':[], 'toilet':[], 'sink':[], 'bathtub':[],
             'garbagebin':[]}
    type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'showercurtrain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'garbagebin':17}  
    
    class2type = {type2class[t]:t for t in type2class}
    nyu40ids = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
    nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(nyu40ids))}
    
    data_infor = {}
    
    for scene_id in tqdm.tqdm(TRAIN_SCAN_NAMES):
        scene_name = 'scannet_train_detection_data/{}'.format(scene_id)
        output_folder = 'data_viz_dump'

        data = np.load(scene_name+'_vert.npy')
        scene_points = data[:,0:3]
        colors = data[:,3:]
        # instance_labels = np.load(scene_name+'_ins_label.npy')
        # semantic_labels = np.load(scene_name+'_sem_label.npy')
        instance_bboxes = np.load(scene_name+'_bbox.npy')
        if len(instance_bboxes) > 0:
            for box in instance_bboxes:
                label_3d = class2type[nyu40id2class[box[-1]]]
                # print(box)
                # print(label_3d)
                single_room[label_3d].append(box[:6].tolist())

            data_infor.update({'{}'.format('{}'.format(scene_id)):single_room})
        else:
            data_infor.update({'{}'.format('{}'.format(scene_id)):[]})
    
    dict2json('./scannet.json', data_infor)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def analyse_box():
    Detec_path = '/media/xhm/Elements/ScanNet_Vis/Scannet_Detection_Instruction_v4_normalized/1400/10_2560_30/vicuna_13b_v0_4gpu_use_enc_ft_v_detr_hiddenlayer_-3_six_token_embedding/Detection_ScanNet_v3.json'
    Detec_GT_path = './Detection_ScanNet_v3.json'
    Detec_save_path = './Detection_ScanNet_v3_gt_location.json'
    
    with open(Detec_GT_path) as f:
        GT = json.load(f)
    
    with open(Detec_path) as f:
        predict = json.load(f)

    print('GT:',len(GT))
    print('prediction:',len(predict))
    
    # assert 1==2
    # gt_count_1 = 0
    # gt_count_2 = 0
    # gt_count_3 = 0
    # gt_count_4 = 0
    
    # predict_count_1 = 0
    # predict_count_2 = 0
    # predict_count_3 = 0
    # predict_count_4 = 0
    
    predict_count = []
    gt_count = []
    map_50_box = []
    map_25_box = []
    normalized_gt_box = []
    predict_list = []
    tabr = tqdm.tqdm(len(GT))
    for i, gt in enumerate(GT):
        single_scene_box = []
        number = i # 40 20  86 pianyi  96 108 120 # training 3 5 8 10 12 14 18 24
        test_scane = GT[number]
        predict_scene = predict[number]
        bboxes = parse_bbox_3d_decode_location(predict_scene['text']) # 从predict中提取出需要的box
        
        # print('length of predict bboxes:', len(bboxes))
        
        # print('length of ground truth:', len(test_scane['object']))
        
        id = test_scane['id']
        
        # if len(bboxes) < 10:
        #     predict_count_1 = predict_count_1 + 1
        # elif len(bboxes) < 20:
        #     predict_count_2 = predict_count_2 + 1
        # elif len(bboxes) < 30:
        #     predict_count_3 = predict_count_3 + 1
        # else:
        #     predict_count_4 = predict_count_4 + 1
        
        # if len(test_scane['object']) < 10:
        #     gt_count_1 = gt_count_1 + 1
        # elif len(test_scane['object']) < 20:
        #     gt_count_2 = gt_count_2 + 1
        # elif len(test_scane['object']) < 30:
        #     gt_count_3 = gt_count_3 + 1
        # else:
        #     gt_count_4 = gt_count_4 + 1
        predict_count.append(len(bboxes))
        gt_count.append(len(test_scane['object']))
        
        count_map25 = 0
        count_map50 = 0
        for gt_box in test_scane['object']:
            
            label = gt_box['label']
            
            gt_bbox = [(gt_box['bbox'][0]+gt_box['bbox'][3])/2, (gt_box['bbox'][1]+gt_box['bbox'][4])/2, (gt_box['bbox'][2]+gt_box['bbox'][5])/2, 
                       (gt_box['bbox'][3]-gt_box['bbox'][0]),   (gt_box['bbox'][4]-gt_box['bbox'][1]),   (gt_box['bbox'][5]-gt_box['bbox'][2])]
            

            normalized_box = [sigmoid(x) for x in gt_bbox]
            
            
            location_gt = [f'<patch_index_{str(int(x*1000)).zfill(4)}>' for x in normalized_box]
            
            location_gt.append(classes_token_map[label][0])
            location_gt = ''.join(location_gt)
            single_scene_box.append(location_gt)
            
            print('ground truth:',gt_bbox)
            for pred_box in bboxes:
                iou = cal_iou_3d(gt_bbox, [pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_box[5], pred_box[4]])
                if iou >= 0.25:
                    print('prediction box',[pred_box[0], pred_box[1], pred_box[2], pred_box[3], pred_box[5], pred_box[4]])
                    count_map25 = count_map25 + 1
                    # print("GT",gt_bbox)
                    # print("Pred", pred_box)
                    # print('IoU',iou)
                elif iou > 0.5:
                    count_map50 = count_map50 + 1
        
        # print(normalized_gt_box)
        
        normalized_gt_box.append(single_scene_box)
                
        map_50_box.append(count_map50)
        map_25_box.append(count_map25)
        
        tabr.update(1)
    
    with open(Detec_save_path,'w+') as f:
        list = json.dumps(normalized_gt_box,indent=4)
        f.write(list)
    
    chosen_num = 312
    count_split = [x for x in range(chosen_num)]
    # 国家的序号0-4
    country_num = np.arange(len(count_split))
    
    # predict_count = [predict_count_1, predict_count_2, predict_count_3, predict_count_4]
    # gt_count = [gt_count_1, gt_count_2, gt_count_3, gt_count_4]
    
    # 设置每个条形的宽度
    width = 0.2
    start_1 = country_num
    start_2 = country_num + width
    start_3 = country_num + width*2
    start_4 = country_num + width*3
    
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         name='val',
    #         x=count_split,
    #         y=gt_count,
    #         mode="markers+lines+text",
    #         text=gt_count,                 
    #         textposition="top center"       
    # ))
    # fig.add_trace(
    #     go.Scatter(
    #         name='prediction',
    #         x=count_split,
    #         y=predict_count,
    #         mode="markers+lines+text",
    #         text=predict_count,                 
    #         textposition="top center"       
    # ))
    # fig.update_layout(
    #     title="Val Set and Prediction Bboxes count",
    #     xaxis=dict(title="scene num", nticks=13),
    #     yaxis=dict(title="count", nticks=11, rangemode="tozero", range=(0,50)),
    #     width=3000,
    #     height=1000
    # )
    # fig.show()
    # pyo.plot(fig, filename='lineplot.jpg')

    # plt.figure(figsize=(50, 20),dpi=100)
    # plt.plot(count_split,gt_count[:chosen_num])
    # plt.plot(count_split,predict_count[:chosen_num])
    # plt.plot(count_split,map_25_box[:chosen_num])
    # plt.plot(count_split,map_50_box[:chosen_num])
    
    # plt.xlabel("scene number",fontsize=14)
    # plt.ylabel("box number ", fontsize=14)
    
    # plt.tick_params(axis="both",labelsize=14)
    
    # plt.legend(labels=['GT Number','Predict Number', 'map25', 'map50'],loc=2)#图例展示位置，数字代表第几象限
    # for i in country_num:
    #     plt.text(start_1[i], gt_count[i], gt_count[i], 
    #             fontdict=dict(fontsize=14),va='bottom',ha='center')
    #     plt.text(start_2[i], predict_count[i], predict_count[i], 
    #             fontdict=dict(fontsize=14),va='bottom',ha='center')
    #     plt.text(start_3[i], map_25_box[i], map_25_box[i], 
    #             fontdict=dict(fontsize=14),va='bottom',ha='center')
    #     plt.text(start_4[i], map_50_box[i], map_50_box[i], 
    #             fontdict=dict(fontsize=14),va='bottom',ha='center')
    # plt.savefig('./count.jpg')
    # plt.show()

    # 直方图
    # trace0 = go.Bar(
    #     x = count_split,
    #     y = gt_count,
    #     name = 'Primary Product',
    #     text=gt_count,
    #     marker=dict(
    #         color = 'rgb(49,130,189)'
    #     )
    # )
    # trace1 = go.Bar(
    #     x = count_split,
    #     y = predict_count,
    #     name = 'Secondary Product',
    #     text=predict_count,
    #     marker=dict(
    #         color = 'rgb(204,204,204)'
    #     )
    # )
    # data = [trace0,trace1]
    # pyo.iplot(data)

    # # 加上label标签，为设置图例使用
    # plt.bar(silver_start, gt_count, width=width,color='silver',label='银牌')
    # plt.bar(gold_start, predict_count, width=width,color='gold',label='金牌')
    # plt.xlabel('国家', fontsize=14,loc='right')
    # plt.ylabel('奖牌数', fontsize=14,loc='top')
    # plt.title('冬奥会各个国家所获奖牌数',fontsize=18)
    # # 为了让国家名称在三个柱状的中间，需要改变xticks
    # plt.xticks(gold_start,count_split,fontsize=14)
    # # 设置图例
    # plt.legend()
    # for i in country_num:
    #     plt.text(gold_start[i], gt_count[i], gt_count[i], 
    #             fontdict=dict(fontsize=14),va='bottom',ha='center')
    #     plt.text(silver_start[i], predict_count[i], predict_count[i], 
    #             fontdict=dict(fontsize=14),va='bottom',ha='center')
    # plt.show()


    
if __name__ == '__main__':
    # creat_scannet_json() # 将数据写进json文件里面
    # visualize_detection_box() # 可视化Votenet处理之后的box数据
    # visualize_scanrefer_benchmark_box() # 可视化scanrefer-benchmark数据
    visualize_scannet_benchmark_box() # 可视化detection-benchmark数据
    # visualize_scanrefer_finetune_box() # 可视化scanrefer-finetune数据
    # visualize_scannet_finetune_box() # 可视化scannet-finetune数据
    # analyse_box() # 统计box的质量
