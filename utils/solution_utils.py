import numpy as np
from shapely.geometry import Polygon
from utils.diffuser_utils import convert_state_to_state_and_action, unicyle_forward_dynamics
from tbsim.utils.geometry_utils import transform_points_tensor, transform_yaw, transform_agents_to_world
import torch
import tbsim.dynamics as dynamics
import math
import torch.nn as nn

VEH_COLL_THRESH = 0.05

def transform2frame(frame, poses, inverse=False):
    '''
    Transform the given poses into the local frame of the given frame.
    All inputs are in the global frame unless inverse=True.
    :param frame: B x 3 where each row is (x, y, h) or B x 4 with (x,y,hx,hy) i.e. heading as a vector
    :param poses: to transform (B x N x 3) or (B x N x 4)
    :param inverse: if true, poses are assumed already in the local frame of frame, and instead transforms
                    back to the global frame based on frame.
    :return: poses (B x N x 3) or (B x N x 4), but in the local frame
    '''
    B, N, D = poses.size()

    # build rotation matrices
    # for frame
    if D == 3:
        # get from angle
        frame_hcos = frame[:, 2].cos()
        frame_hsin = frame[:, 2].sin()
    else:
        frame_hcos = frame[:, 2]
        frame_hsin = frame[:, 3]
    Rf = torch.stack([frame_hcos, frame_hsin, -frame_hsin, frame_hcos], dim=1)
    Rf = Rf.reshape((B, 1, 2, 2)).expand(B, N, 2, 2)
    # and for poses
    if D == 3:
        # get from angle
        poses_hcos = poses[:, :, 2].cos()
        poses_hsin = poses[:, :, 2].sin()
    else:
        poses_hcos = poses[:, :, 2]
        poses_hsin = poses[:, :, 3]
    Rp = torch.stack([poses_hcos, -poses_hsin, poses_hsin, poses_hcos], dim=2)
    Rp = Rp.reshape((B, N, 2, 2))

    # compute relative rotation
    if inverse:
        Rp_local = torch.matmul(Rp, Rf.transpose(2, 3))
    else:
        Rp_local = torch.matmul(Rp, Rf)
    local_hcos = Rp_local[:, :, 0, 0]
    local_hsin = Rp_local[:, :, 1, 0]
    if D == 3:
        local_h = torch.atan2(local_hsin, local_hcos)
        local_h = local_h.unsqueeze(-1)
    else:
        local_h = torch.stack([local_hcos, local_hsin], dim=2)

    # now translation
    frame_t = frame[:, :2].reshape((B, 1, 2))
    poses_t = poses[:, :, :2]
    if inverse:
        local_t = torch.matmul(Rf.transpose(2, 3), poses_t.reshape((B, N, 2, 1)))[:, :, :, 0]
        local_t = local_t + frame_t
    else:
        local_t = poses_t - frame_t
        local_t = torch.matmul(Rf, local_t.reshape((B, N, 2, 1)))[:, :, :, 0]

    # all together
    local_poses = torch.cat([local_t, local_h], dim=-1)
    return local_poses

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2]
    return simple_box

def check_vehicle_collision(ego_traj, ego_lw, agents_traj, agents_lw):
    '''
    ego_traj: T x 4
    ego_lw: 2
    agents_traj: N x T x 4
    agents_lw: N x 2
    '''
    NA, FT, _ = agents_traj.shape
    veh_coll = np.zeros((NA), dtype=np.bool_)
    coll_time = np.ones((NA), dtype=np.int16) * FT
    poly_cache = dict() # for the tgt polygons since used many times
    for aj in range(NA):
        for t in range(FT):
            # compute iou
            if t not in poly_cache:
                ai_state = ego_traj[t, :]
                ai_corners = get_corners(ai_state, ego_lw)
                ai_poly = Polygon(ai_corners)
                poly_cache[t] = ai_poly
            else:
                ai_poly = poly_cache[t]

            aj_state = agents_traj[aj, t, :]
            aj_corners = get_corners(aj_state, agents_lw[aj])
            aj_poly = Polygon(aj_corners)
            cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
            if cur_iou > VEH_COLL_THRESH:
                veh_coll[aj] = True
                coll_time[aj] = t
                break # don't need to check rest of sequence

    return veh_coll, coll_time

def convert_action_to_state_and_action(x_out, curr_states, dt=0.1, dyn=None):
    '''
    Apply dynamics on input action trajectory to get state+action trajectory
    Note:
        Support both agent-centric and scene-centric (extra dimension for the inputs).
    Input:
        x_out: (B, (M), T, 2). scaled action trajectory
        curr_states: (B, (M), 4). current state
    Output:
        x_out: (B, (M), T, 6). scaled state+action trajectory
    '''
    x_out_state = unicyle_forward_dynamics(
        dyn_model=dyn,
        initial_states=curr_states,
        actions=x_out,
        step_time=dt,
        mode='parallel',
    )

    x_out_all = torch.cat([x_out_state, x_out], dim=-1)

    return x_out_all

def run_find_solution_optim(buffer, ego_traj, agents_traj, ego_lw, agents_lw, crash_time, crash_agent, first_obs):
    '''
    ego_traj: T x 4
    ego_lw: 2
    agents_traj: N x T x 4
    agents_lw: N x 2
    '''
    ego_traj_tensor = torch.from_numpy(ego_traj).cuda()
    # convert ego_traj to x, y, yaw
    ego_traj_tensor = torch.cat([ego_traj_tensor[:, :2], torch.arctan2(ego_traj_tensor[:, 3], ego_traj_tensor[:, 2]).unsqueeze(1)], dim=1)
    agent_from_world = torch.inverse(torch.from_numpy(buffer['world_from_agent'][0, 0]).cuda())
    ego_pos = transform_points_tensor(ego_traj_tensor[:, :2], agent_from_world)
    ego_yaw = transform_yaw(ego_traj_tensor[:, 2], agent_from_world.unsqueeze(0).repeat(ego_traj_tensor.shape[0], 1, 1)).unsqueeze(1)
    ego_traj_tensor = torch.cat([ego_pos, ego_yaw], dim=1)

    ego_state_and_action = convert_state_to_state_and_action(ego_traj_tensor[1: crash_time + 10], first_obs['agents']['curr_speed'][0], dt=0.1).detach() # 需要agent-centric
    if not ego_state_and_action.requires_grad:
        ego_state_and_action.requires_grad_()
    opt = torch.optim.Adam([ego_state_and_action], lr=0.05)
    adv_loss = AdversarialLoss()
    data_batch = {
            'world_from_agent': torch.from_numpy(buffer['world_from_agent'][0, 0]).cuda(),
            'other_agent_traj': torch.from_numpy(agents_traj).cuda()[:, 1: crash_time + 10],
            'crash_agent': crash_agent,
        }

    dyn = dynamics.Unicycle(
            "dynamics",
            max_steer=0.5,
            max_yawvel=math.pi * 2.0,
            acce_bound=(-10, 8)
        )
    for _ in range(200):
        with torch.enable_grad():
            curr_states = torch.cat([ego_pos[0], first_obs['agents']['curr_speed'][0].unsqueeze(0), ego_yaw[0]], dim=-1).unsqueeze(0)
            
            x_loss = convert_action_to_state_and_action(ego_state_and_action[..., 4:].unsqueeze(0), curr_states, dt=0.1, dyn=dyn)
            
            loss = adv_loss(x_loss, data_batch)
        
        loss.backward()
        opt.step()
        opt.zero_grad()

    
    x_final = convert_action_to_state_and_action(ego_state_and_action[..., 4:].unsqueeze(0), curr_states, dt=0.1, dyn=dyn)
    pos_pred_global, yaw_pred_global = transform_agents_to_world(
        x_final[..., :2].unsqueeze(0), x_final[..., 3:4].unsqueeze(0), data_batch['world_from_agent'])
    return pos_pred_global.reshape(-1, 2).detach().cpu().numpy(), yaw_pred_global.reshape(-1).detach().cpu().numpy()


class AdversarialLoss(nn.Module):
    def __init__(self, guide_moving_speed_th=5e-1, sol_min_t=0):
        super().__init__()
        self.guide_moving_speed_th = guide_moving_speed_th
        self.sol_min_t = sol_min_t
    
    def forward(self, x, data_batch):
        '''
        x : 1 x T x 6
        '''
        data_world_from_agent = data_batch["world_from_agent"]
        x_baseline = data_batch.get('x_baseline', None)
        if x_baseline is None:
            x_baseline = x.detach().clone()
            data_batch['x_baseline'] = x_baseline
        # mask unmoving agents

        pos_pred_global, yaw_pred_global = transform_agents_to_world(x[..., :2].unsqueeze(0), x[..., 3:4].unsqueeze(0), data_world_from_agent)

        # 1. find the adversary which is the closest agent to ego
        ego_pos = pos_pred_global[0, 0, self.sol_min_t:, :] 
        # ego_yaw = yaw_pred_global[0, :, self.crash_min_t:, :] 
        agent_pos = data_batch['other_agent_traj'][:, self.sol_min_t:, :2]
        crash_agent = data_batch['crash_agent']
        crash_agent_pos = agent_pos[crash_agent]
        dist_traj = torch.norm(ego_pos - crash_agent_pos, dim=-1)
        min_dist_in  = dist_traj

        T = min_dist_in.size(-1) - self.sol_min_t
        min_dist = nn.functional.softmin(min_dist_in.reshape(-1), dim=-1)
        max_dist = nn.functional.softmax(min_dist_in.reshape(-1), dim=-1)

        adv_loss = -((dist_traj.reshape(-1) ** 2) * min_dist).sum(-1)
        
        # 3. normalize the loss
        normal_loss = ((torch.norm(x[..., :2].squeeze(0) - x_baseline[..., :2].squeeze(0), dim=-1) ** 2)).sum(-1) \
            + ((x[..., 3] - x_baseline[..., 3]) ** 2).sum(-1)
        
        return adv_loss + 0.5 * normal_loss
