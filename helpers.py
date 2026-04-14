import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import torch
import copy

def cart_2_spherical(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    theta = theta % (2 * np.pi)
    arg = z/r
    phi = np.arcsin(arg)
    return r, theta, phi

# def plot_trace(trace):
#     x = trace[:, 0]
#     y = trace[:, 1]
#     z = trace[:, 2]
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     # Plot the data points
#     ax.scatter(x, y, z) 

#     # Label the axes
#     ax.set_xlabel('X Axis')
#     ax.set_ylabel('Y Axis')
#     ax.set_zlabel('Z Axis')
#     ax.set_title('Sparse object example')
#     # --- key part ---
#     dx = x.max() - x.min()
#     dy = y.max() - y.min()
#     dz = z.max() - z.min()

#     ax.set_box_aspect((dx, dy, dz))  # rectangular prism
#     # -----------------

#     plt.show()


def plot_trace(trace):
    pio.renderers.default = "iframe_connected"
    x = trace[:, 0]
    y = trace[:, 1]
    z = trace[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, 
        y=y, 
        z=z,
        mode='markers',
        marker=dict(
            size=3,              # Adjust size for visibility
            color=z,             # Optional: Color by height (Z) for depth perception
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    # Update layout to match your matplotlib logic
    fig.update_layout(
        title='Sparse object example',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            # --- Key Part ---
            # aspectmode='data' automatically handles the dx, dy, dz calculation 
            # you were doing manually. It ensures 1 unit in X looks the same 
            # physical length as 1 unit in Y or Z.
            aspectmode='data' 
            # ----------------
        ),
        margin=dict(r=0, l=0, b=0, t=40) # Tight layout
    )

    fig.show()
    
    
    
def plot_frame(data_dict, dataset, spoof_frame, start_idx, end_idx, max_points):
    pio.renderers.default = "iframe_connected"

    # Animate frames 0-197 from the dataset (same sequence assumed)
    # start_idx = 0
    # end_idx = 1
    # max_points = 150000  # downsample per frame for speed; raise/lower as needed
    rng = np.random.default_rng(0)

    frame_points = []
    mins = []
    maxs = []

    for i in range(start_idx, end_idx + 1):
        # info = dataset.infos[i]
        # seq = info['point_cloud']['lidar_sequence']
        # sample_idx = info['point_cloud']['sample_idx']
        if(i == spoof_frame):
            pts = data_dict['points'][:, :3]
        else:
            pts = dataset[i]['points'][:, :3]
            
        print(i, pts.shape)
        if pts.shape[0] > max_points:
            sel = rng.choice(pts.shape[0], size=max_points, replace=False)
            pts = pts[sel]

        frame_points.append(pts)
        mins.append(pts.min(axis=0))
        maxs.append(pts.max(axis=0))

    global_min = np.min(np.vstack(mins), axis=0)
    global_max = np.max(np.vstack(maxs), axis=0)

    def make_scatter(pts):
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=1, color=pts[:, 2], colorscale='Viridis', opacity=0.6),
        )

    frames = [
        go.Frame(data=[make_scatter(pts)], name=str(i))
        for i, pts in zip(range(start_idx, end_idx + 1), frame_points)
    ]


    fig = go.Figure(
        data=[make_scatter(frame_points[0])],
        frames=frames,
    )

    fig.update_layout(
        title=f'Waymo sequence frames {start_idx}-{end_idx}',
        scene=dict(
            xaxis=dict(range=[global_min[0], global_max[0]], title='x'),
            yaxis=dict(range=[global_min[1], global_max[1]], title='y'),
            zaxis=dict(range=[global_min[2], global_max[2]], title='z'),
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'x': 0.1,
            'y': 0,
            'pad': {'r': 10, 't': 70},
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                },
            ],
        }],
        sliders=[{
            'x': 0.1,
            'y': 0,
            'len': 0.9,
            'steps': [
                {
                    'method': 'animate',
                    'args': [[str(i)], {'frame': {'duration': 5, 'redraw': True}, 'mode': 'immediate'}],
                    'label': str(i),
                }
                for i in range(start_idx, end_idx + 1)
            ],
        }],
    )

    fig.show()
    
    
#     azimuth_elevation_indices_trace = []
# dtheta = np.deg2rad(0.1358)
# grid = np.zeros((len(azimuths), len(elevations)))
# for i in range (0, len(trace_spherical)):
#     # Your specific value
#     query_value = trace_spherical[i][2] 

#     # Calculate absolute difference between the query and ALL elevation values
#     differences = np.abs(elevations - query_value)

#     # Find the index of the smallest difference
#     closest_index = np.argmin(differences)

#     # print(f"Closest Index: {closest_index}")
#     # print(f"Value at Index: {elevations[closest_index]}")
#     # print(np.floor((trace_spherical[i][1] % (2*np.pi)) / dtheta), closest_index)
#     azimuth_elevation_indices_trace.append((np.floor((trace_spherical[i][1] % (2*np.pi)) / dtheta), closest_index))

# 1. Define mapping helper
# Waymo labels: 1=Vehicle, 2=Pedestrian, 3=Cyclist
# dataset.class_names is typically ['Vehicle', 'Pedestrian', 'Cyclist']
def inject_gt_names(data_dict, class_names):
    if 'gt_boxes' in data_dict and 'gt_names' not in data_dict:
        # Extract the last column (label index)
        # gt_boxes shape is (N, 8), index 7 is the label
        labels = data_dict['gt_boxes'][:, -1].astype(int)
        
        # Map index to name (Label 1 -> Index 0)
        # We use l-1 because Waymo labels are 1-based
        names = np.array([class_names[l - 1] for l in labels])
        data_dict['gt_names'] = names
    return data_dict

def convert_to_batch(dict_):
    data_dict = copy.deepcopy(dict_)
    device = dict_['points'].device
    data_dict['sample_idx'] = torch.tensor([dict_['sample_idx']], device = device)
    rows_points = dict_['points'].shape[0]
    zeros_points =torch.zeros((rows_points, 1), device = device)
    data_dict['points'] = torch.cat((zeros_points, dict_['points']), dim = 1)
    data_dict['frame_id'] = [dict_['frame_id']]
    data_dict['gt_boxes'] = dict_['gt_boxes'].unsqueeze(0)
    data_dict['lidar_aug_matrix'] = dict_['lidar_aug_matrix'].unsqueeze(0)
    data_dict['use_lead_xyz'] = torch.tensor([1], device = device) if dict_['use_lead_xyz'] else torch.tensor([0], device = device)
    rows_vc = dict_['voxel_coords'].shape[0]
    zeros_vc = torch.zeros((rows_vc, 1), device = device)
    data_dict['voxel_coords'] = torch.cat((zeros_vc, dict_['voxel_coords']), dim = 1)
    data_dict['metadata'] = [dict_['metadata']]
    
    return data_dict

def cosine_similarity(a, b, eps=1e-8):
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + eps)