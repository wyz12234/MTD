import numpy as np
import torch
from tbsim.dynamics.base import Dynamics
import math
import time

class Progress:

	def __init__(self, total, name = 'Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

		
	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
					(key, val)
					for key, val in params.items()
				])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines


		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')
		
	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent*100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i+n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0,'')
		padding = '\n' + ' '*self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()

@torch.no_grad()
def ubound(dyn_model, v):
    yawbound = torch.minimum(
        dyn_model.max_steer * torch.abs(v),
        dyn_model.max_yawvel / torch.clip(torch.abs(v), min=0.1),
    )
    yawbound = torch.clip(yawbound, min=0.1)
    acce_lb = torch.clip(
        torch.clip(dyn_model.vbound[0] - v, max=dyn_model.acce_bound[1]),
        min=dyn_model.acce_bound[0],
    )
    acce_ub = torch.clip(
        torch.clip(dyn_model.vbound[1] - v, min=dyn_model.acce_bound[0]),
        max=dyn_model.acce_bound[1],
    )
    lb = torch.cat((acce_lb, -yawbound), dim=-1)
    ub = torch.cat((acce_ub, yawbound), dim=-1)
    return lb, ub

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def torch_bmm(a, b):
    if a.ndim == 3:
        return torch.bmm(a, b)
    elif a.ndim == 4:
        return torch.einsum('bijk,bikl->bijl', a, b)

def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (..., 1)
    :param theta2: angle 2 (..., 1)
    :return diff: smallest angle difference between angles (..., 1)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def convert_state_to_state_and_action(traj_state, vel_init, dt, data_type='torch'):
    '''
    Infer vel and action (acc, yawvel) from state (x, y, yaw) based on Unicycle.
    Note:
        Support both agent-centric and scene-centric (extra dimension for the inputs).
    Input:
        traj_state: (batch_size, [num_agents], num_steps, 3)
        vel_init: (batch_size, [num_agents],)
        dt: float
        data_type: ['torch', 'numpy']
    Output:
        traj_state_and_action: (batch_size, [num_agents], num_steps, 6)
    '''
    BM = traj_state.shape[:-2]
    if data_type == 'torch':
        sin = torch.sin
        cos = torch.cos

        # device = traj_state.get_device()
        device = traj_state.device
        pos_init = torch.zeros(*BM, 1, 2, device=device)
        yaw_init = torch.zeros(*BM, 1, 1, device=device)
    elif data_type == 'numpy':
        sin = np.sin
        cos = np.cos

        pos_init = np.zeros((*BM, 1, 2))
        yaw_init = np.zeros((*BM, 1, 1))
    else:
        raise
    def cat(arr, dim):
        if data_type == 'torch':
            return torch.cat(arr, dim=dim)
        elif data_type == 'numpy':
            return np.concatenate(arr, axis=dim)

    target_pos = traj_state[..., :2]
    traj_yaw = traj_state[..., 2:]    

    # pre-pad with zero pos and yaw
    pos = cat((pos_init, target_pos), dim=-2)
    yaw = cat((yaw_init, traj_yaw), dim=-2)

    # estimate speed from position and orientation
    vel_init = vel_init[..., None, None]
    vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * cos(
        yaw[..., 1:, :]
    ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * sin(
        yaw[..., 1:, :]
    )
    vel = cat((vel_init, vel), dim=-2)
    
    # m/s^2
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    # rad/s
    # yawvel = (yaw[..., 1:, :] - yaw[..., :-1, :]) / dt
    yawdiff = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :])
    yawvel = yawdiff / dt
    
    pos, yaw, vel = pos[..., 1:, :], yaw[..., 1:, :], vel[..., 1:, :]

    traj_state_and_action = cat((pos, vel, yaw, acc, yawvel), dim=-1)

    return traj_state_and_action

def unicyle_forward_dynamics(
    dyn_model: Dynamics,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    step_time: float,
    mode: str = 'parallel',
):
    """
    Integrate the state forward with initial state x0, action u
    Note:
        Support both agent-centric and scene-centric (extra dimension for the inputs).
    Args:
        dyn_model (dynamics.Dynamics): dynamics model
        initial_states (Torch.tensor): state tensor of size [B, (A), 4]
        actions (Torch.tensor): action tensor of size [B, (A), T, 2]
        step_time (float): delta time between steps
        mode (str): 'parallel' or 'partial_parallel' or 'chain'. 'parallel' is the fastet
        but it generates different results from 'partial_parallel' and 'chain' when the
        velocity is out of bounds.
        
        When running one (three) inner loop gradient update, the network related time for each are:
        parallel: 1.2s (2.5s)
        partial_parallel: 2.9s (7.0s)
        chain: 4.4s (10.4s)
        original implementation: 5.8s (14.6s)

    Returns:
        state tensor of size [B, (A), T, 4]
    """
    

    # ------------------------------------------------------------ #
    if mode in ['parallel', 'partial_parallel']:
        with torch.no_grad():
            num_steps = actions.shape[-2]
            bm = actions.shape[:-2]
            device = initial_states.device

            mat = torch.ones(num_steps+1, num_steps+1, device=device)
            mat = torch.tril(mat)
            mat = mat.repeat(*bm, 1, 1)
            
            mat2 = torch.ones(num_steps, num_steps+1, device=device)
            mat2_h = torch.tril(mat2, diagonal=1)
            mat2_l = torch.tril(mat2, diagonal=-1)
            mat2 = torch.logical_xor(mat2_h, mat2_l).float()*0.5
            mat2 = mat2.repeat(*bm, 1, 1)

        acc = actions[..., :1]
        yawvel = actions[..., 1:]
        
        acc_clipped = torch.clip(acc, dyn_model.acce_bound[0], dyn_model.acce_bound[1])
        
        if mode == 'parallel':
            acc_paded = torch.cat((initial_states[..., -2:-1].unsqueeze(-2), acc_clipped*step_time), dim=-2)
            v_raw = torch_bmm(mat, acc_paded)
            v_clipped = torch.clip(v_raw, dyn_model.vbound[0], dyn_model.vbound[1])
        else:
            v_clipped = [initial_states[..., 2:3]] + [None] * num_steps
            for t in range(num_steps):
                vt = v_clipped[t]
                acc_clipped_t = torch.clip(acc_clipped[..., t], dyn_model.vbound[0] - vt, dyn_model.vbound[1] - vt)
                v_clipped[t+1] = vt + acc_clipped_t * step_time
            v_clipped = torch.stack(v_clipped, dim=-2)
            
        v_avg = torch_bmm(mat2, v_clipped)
        
        v = v_clipped[..., 1:, :]

        with torch.no_grad():
            v_earlier = v_clipped[..., :-1, :]
            yawbound = torch.minimum(
                dyn_model.max_steer * torch.abs(v_earlier),
                dyn_model.max_yawvel / torch.clip(torch.abs(v_earlier), min=0.1),
            )
            yawbound_clipped = torch.clip(yawbound, min=0.1)
        
        yawvel_clipped = torch.clip(yawvel, -yawbound_clipped, yawbound_clipped)

        yawvel_paded = torch.cat((initial_states[..., -1:].unsqueeze(-2), yawvel_clipped*step_time), dim=-2)
        yaw_full = torch_bmm(mat, yawvel_paded)
        yaw = yaw_full[..., 1:, :]

        # print('before clip', torch.cat((acc[0], yawvel[0]), dim=-1))
        # print('after clip', torch.cat((acc_clipped[0], yawvel_clipped[0]), dim=-1))

        yaw_earlier = yaw_full[..., :-1, :]
        vx = v_avg * torch.cos(yaw_earlier)
        vy = v_avg * torch.sin(yaw_earlier)
        v_all = torch.cat((vx, vy), dim=-1)

        # print('initial_states[0, -2:]', initial_states[0, -2:])
        # print('vx[0, :5]', vx[0, :5])

        v_all_paded = torch.cat((initial_states[..., :2].unsqueeze(-2), v_all*step_time), dim=-2)
        x_and_y = torch_bmm(mat, v_all_paded)
        x_and_y = x_and_y[..., 1:, :]

        x_all = torch.cat((x_and_y, v, yaw), dim=-1)
    
    # ------------------------------------------------------------ #
    elif mode == 'chain':
        num_steps = actions.shape[-2]
        x_all = [initial_states] + [None] * num_steps
        for t in range(num_steps):
            x = x_all[t]
            u = actions[..., t, :]
            dt = step_time
            
            with torch.no_grad():
                lb, ub = ubound(dyn_model, x[..., 2:3])
            # print('chain before clip u[0]', u[0])
            u = torch.clip(u, lb, ub)
            # print('chain after clip u[0]', u[0])
            theta = x[..., 3:4]
            dxdt = torch.cat(
                (
                    torch.cos(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    torch.sin(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    u,
                ),
                dim=-1,
            )
            # print('x[0, :3]', x[0, :3])
            # print(t, 'dxdt[0, 0]', dxdt[0, 0])
            x_all[t + 1] = x + dxdt * dt
        x_all = torch.stack(x_all[1:], dim=-2)
    # ------------------------------------------------------------ #
    else:
        raise

    return x_all