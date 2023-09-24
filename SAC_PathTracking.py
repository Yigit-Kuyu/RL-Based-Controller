import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import time
from collections import deque
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import math


WB = 0.1    # WB: The wheelbase of the vehicle, which is the distance between the front and rear axles.                                                 [Pure Pursuit]
dt = 0.1    # dt: The time step used in the simulation, representing the interval at which updates are calculated. # 0.1                                [PID Speed Control]
k = 0.1     # k: This is the look forward gain, which scales the look-ahead distance based on the vehicle's velocity.                                   [Pure Pursuit]
Lfc = 2.5   # Lfc: The look-ahead distance, which determines how far ahead the vehicle should consider when choosing the target point on the path.      [Pure Pursuit]
Kp = 1.0    # Kp: The speed proportional gain, used in the proportional control law to adjust the acceleration.                                         [PID Speed Control]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class InitialState_Update:

    """

    This class represents the state of the vehicle, including its position (x, y), orientation (yaw), and velocity (v).
    The __init__ method initializes the state and calculates the rear axle position (rear_x, rear_y) based on the given wheelbase (WB).
    The update method is used to update the state based on control inputs (a for acceleration and delta for steering angle).
    The calc_distance method calculates the distance between the rear axle of the vehicle and a given point (point_x, point_y). 

    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):

        self.x = x                                                  # x (float): The x position of the vehicle.
        self.y = y                                                  # y (float): The y position of the vehicle.
        self.yaw = yaw                                              # yaw (float): The orientation of the vehicle.
        self.v = v                                                  # v (float): The velocity of the vehicle.
        
        # Positions of rear wheels
        
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))      # rear_x (float): The x position of the rear axle.
                                                                    # X =x + L*cos(ϴ) [ref:3. pp:3. eq:4]

        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))      # rear_y (float): The y position of the rear axle.
                                                                    #  Y =y + L* sin(ϴ) [ref:3. pp:3. eq:2b ]
    def update(self, a, delta):

        """ 
        The aim of this section is to simulate how the vehicle's position, orientation, and velocity change over a small time interval in response to the applied control inputs. 
        This simulation is essential for understanding how the vehicle behaves as it follows the desired trajectory and adjusts its speed and steering to stay on track.

        """

        self.x += self.v * math.cos(self.yaw) * dt                  # x (float): The x position of the vehicle. Updates the vehicle's x position by integrating its velocity (self.v) along the x-axis,
                                                                    # taking into account its orientation (self.yaw) to calculate the change in x position over the time interval dt.
                                                                    # X = V * cos(ψ+β) (ref:3 pp:3 eq:13)
                                                                    # vehicle slip angle β = arctan ((lr*tan(δ))/(lf+lr)) (ref:3 pp:3 eq:16)
                                                                    

        self.y += self.v * math.sin(self.yaw) * dt                  # y (float): The y position of the vehicle. Similar to the previous line, 
                                                                    # this updates the vehicle's y position based on its velocity and orientation.
                                                                    # Y = V * sin(ϴ+β) [Lateral Vehicle Dynamics (ref:3 pp:3 eq:9)]
                                                                    # vehicle slip angle β = arctan ((lr*tan(δ))/(lf+lr)) (ref:3 pp:3 eq:16)


        self.yaw += self.v / WB * math.tan(delta) * dt              # yaw (float): The orientation of the vehicle. Updates the vehicle's orientation (yaw) by integrating the change in yaw angle over time. 
                                                                    # The change in yaw angle is calculated based on the vehicle's velocity, the wheelbase (WB), and the calculated steering angle (delta).
                                                                    # ϴ = V * tan(δ) / L [ref:1 pp:3 eq:2c]
                                                                     

        self.v += a * dt                                            # v (float): The velocity of the vehicle. Updates the vehicle's velocity by integrating the acceleration (a) over time.
                                                                    # This accounts for changes in the vehicle's speed due to the applied acceleration.
                                                                    
        
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))      # rear_x (float): The x position of the rear axle. Updates the x position of the rear axle based on the updated vehicle position and yaw angle. 
                                                                    # self.x is the x coordinate of the vehicle's center of gravity, and WB is the wheelbase.
                                                                    # This is used for determining the position of the rear of the vehicle.
                                                                    # X = cos(ϴ) [ref:3. pp:3. eq:8]


        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))      # rear_y (float): The y position of the rear axle.Similar to the previous line,
                                                                    # this updates the y position of the rear axle based on the updated vehicle position and yaw angle.
    
    
    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x                                  # dx (float): The distance between the rear axle and the given point along the x-axis.
        dy = self.rear_y - point_y                                  # dy (float): The distance between the rear axle and the given point along the y-axis.
        return math.hypot(dx, dy)                                                                    # Y = sin(ϴ) [ref:3. pp:3. eq:9]

class StateSave:

    def __init__(self ):                                            # This class is used to store the vehicle's state at each time step.
        self.x = []                                                 # x (list): A list of the vehicle's x position at each time step.
        self.y = []                                                 # y (list): A list of the vehicle's y position at each time step.
        self.yaw = []                                               # yaw (list): A list of the vehicle's orientation at each time step.
        self.v = []                                                 # v (list): A list of the vehicle's velocity at each time step.
        self.t = []                                                 # t (list): A list of the time at each time step.

    def append_all(self, t, state):                                     # This method is used to append the vehicle's state at a given time step to the corresponding lists.
        self.x.append(state.x)                                      # t (float): The time at the given time step.
        self.y.append(state.y)                                      # state (State): The vehicle's state at the given time step.
        self.yaw.append(state.yaw)                                  # This method is used to append the vehicle's state at a given time step to the corresponding lists.
        self.v.append(state.v)                                      # This method is used to append the vehicle's state at a given time step to the corresponding lists.
        self.t.append(t)                                            # This method is used to append the vehicle's state at a given time step to the corresponding lists.





def pure_pursuit_steer_control(state, trajectory, pind,lf_algo):                    # The pure_pursuit_steer_control function implements the pure pursuit steering control algorithm. This algorithm determines the steering angle (delta) that the vehicle should apply to follow a predefined trajectory accurately. 
   
    ind, Lf = trajectory.search_target_index(state,lf_algo)

    """ 
        Target Point Index Calculation:
        This line calls the search_target_index method from the trajectory object to find the index of the target point on the trajectory that the vehicle should aim to follow.
        It also obtains the calculated look-ahead distance Lf for the pure pursuit algorithm.

    """

    if pind >= ind:
        ind = pind
        """ 
            Index Update and Target Point Coordinates:
            This condition ensures that if the input pind (previously provided index) is greater than or equal to the index calculated from the search_target_index, the calculated index (ind) is updated to match pind.
            This allows the algorithm to handle situations where specific points on the trajectory need to be followed.

        """
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]

        """ 
            If the calculated index (ind) is within the range of the trajectory points, the x and y coordinates of the target point are obtained (tx, ty) based on that index.
            If the calculated index exceeds the available trajectory points, it means the vehicle is approaching the end of the trajectory. In this case, the target point is set to the last point of the trajectory (tx, ty).
        """    
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = -math.atan2(-ty + state.rear_y, -tx + state.rear_x) + state.yaw  # [ref:4 pp:3 eq:1]
    """ 
        Alpha Calculation: 
            This line calculates the angle (alpha) between the line connecting the vehicle's rear axle and the target point (tx, ty) and the current orientation of the vehicle (state.yaw).
            It uses the math.atan2 function to ensure the angle is calculated within the correct range.
    """

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0) # δ = arctan(2 * L * sin(α) / ld) [Automatic Steering Methods for Autonomous Automobile Path Tracking (ref:2 pp:17 eq:3)]
        # ".1.0 is just a scaling factor to make delta smaller. "
    """
        Steering Angle Calculation:
            This line calculates the steering angle (delta) that the vehicle should apply to follow the target point.
            It uses the calculated angle alpha and the look-ahead distance Lf to determine the appropriate steering angle based on the pure pursuit algorithm.
            The formula incorporates the vehicle's wheelbase (WB) to convert the angle into a steering angle.
    """

    return delta, ind
    """ 
        Return Results:
            The function returns the calculated steering angle delta and the index of the target point (ind) to be used in the main simulation loop.
            
    """
""" 
    The aim of this section is to calculate the appropriate steering angle for the vehicle to follow the target point on the trajectory.
    The pure pursuit algorithm works by determining the angle required to point the vehicle's front wheels toward the target point while considering the vehicle's kinematics and geometry. 
    This ensures accurate tracking of the desired trajectory.
"""


def proportional_control(target, current):                          # This function calculates the acceleration required to adjust the vehicle's velocity to the target velocity.
    p = Kp * (target - current)                                     # target (float): The target velocity.

    return p  


class TargetCourse:                                                 # This class represents the target course, which is the desired trajectory that the vehicle should follow.

    def __init__(self, cx, cy):                                     # cx (list): A list of the x coordinates of the target course.
        self.cx = cx                                                # cy (list): A list of the y coordinates of the target course.
        self.cy = cy                                                # This method is used to initialize the target course with the given x and y coordinates.
        self.old_nearest_point_index = None                         # old_nearest_point_index (int): The index of the nearest point on the target course to the vehicle's rear axle at the previous time step.

    def search_target_index(self, state,lf_algo):                           # This method is used to find the index of the target point on the target course that the vehicle should follow.

        # To speed up nearest point search, doing it at only first time.
        
        if self.old_nearest_point_index is None: 
           
            dx = [state.rear_x - icx for icx in self.cx]            # dx (list): A list of the distances between the vehicle's rear axle and each point on the target course along the x-axis.
            dy = [state.rear_y - icy for icy in self.cy]            # dy (list): A list of the distances between the vehicle's rear axle and each point on the target course along the y-axis.
            d = np.hypot(dx, dy)                                    # d (list): A list of the distances between the vehicle's rear axle and each point on the target course using the Pythagorean theorem. (c = sqrt(a^2 + b^2)
            ind = np.argmin(d)                                      # ind (int): The index of the nearest point on the target course to the vehicle's rear axle.
            self.old_nearest_point_index = ind                      # This method is used to find the index of the target point on the target course that the vehicle should follow.
            
            """ 
                Initialization and Nearest Point Search:
                Search Nearest Point Index
                This condition checks if the nearest point index has been previously calculated.
                If not, it calculates the distances (dx and dy) between the vehicle's rear axle and each point on the target course along the x-axis and y-axis.
                Using these distances, it calculates the overall distance (d) between the rear axle and each point using the Pythagorean theorem. (c = sqrt(a^2 + b^2)
                It identifies the index of the nearest point (ind) by finding the minimum distance.
                The index is stored in self.old_nearest_point_index for later use.

            """
        else:
            ind = self.old_nearest_point_index                                  # This method is used to find the index of the target point on the target course that the vehicle should follow.
            distance_this_index = state.calc_distance(self.cx[ind], 
                                                      self.cy[ind])             # distance_this_index (float): The distance between the vehicle's rear axle and the point on the target course at the given index.
            while True: 
                if ind+1>=len(self.cx):
                   print('break girdi')
                   break
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])     # distance_next_index (float): The distance between the vehicle's rear axle and the point on the target course at the next index.
                if distance_this_index < distance_next_index: 
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind 
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

            """ 
            Iterative Search for Look-Ahead Point:
            If the nearest point index has been previously calculated, it starts the search from the stored index (ind).
            It calculates the distance (distance_this_index) between the rear axle and the point at the current index.
            It enters a loop that iterates through the trajectory points, comparing distances to find the next point that is farther away.
            If the distance to the next point is greater than the distance to the current point, it means the current point is the nearest among these points. The loop breaks, and ind now points to the nearest point.

            """

        Lf = k * state.v + Lfc  # Look-Ahead Distance Update: The look-ahead distance (Lf) is calculated by scaling the vehicle's velocity (state.v) using the factor k and adding the base look-ahead distance Lfc
        Lf=lf_algo
       
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

            """ 
            Search for Look-Ahead Target Point Index:
            This loop iterates through the trajectory points starting from the previously found index (ind).
            It checks if the look-ahead distance Lf is greater than the distance to the current point on the trajectory.
            If it is, the loop continues to the next point. If it reaches the end of the trajectory or the remaining points don't exceed the look-ahead distance, the loop breaks.

            """

        return ind, Lf

def cross_track_error(cx, cy, x_car, y_car):

  cxy_car=[cx, cy]
  xy_car=[x_car, y_car]
  cxy_car = np.array(cxy_car).T

  cte_list = [np.sqrt((x - xy_car[0])**2 + (y - xy_car[1])**2) for x, y in cxy_car]
  CTE = min(cte_list)

  return CTE

def reward_calculate(state_input):
  CTE=state_input[0]
  lf=state_input[1]
  str=state_input[2]

  R=-0.005*lf+-0.005*str-0.5*(CTE**2)-0.05*abs(CTE)-0.1*dt

  

  if CTE>2:
      done=True
  else:
      done=False

  return R, done





# Use Xavier initialization for the weights and initializes the biases to zero for linear layers.
# It sets the weights to values drawn from a Gaussian distribution with mean 0 and variance
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



class ValueNetwork(nn.Module): # state-Value network
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # Optional


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# stocastic policy
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, high, low):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_size)
        self.log_std = nn.Linear(hidden_dim, action_size)
        
        self.high = torch.tensor(high).to(device)
        self.low = torch.tensor(low).to(device)
        
        self.apply(weights_init_) # Optional
        
        # Action rescaling
        self.action_scale = torch.tensor((high - low) / 2.).to(device)
        self.action_bias = torch.tensor((high + low) / 2.).to(device)
    
    def forward(self, state):
        log_std_min=-20
        log_std_max=2
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        m = self.mean(x)
        s = self.log_std(x)
        s = torch.clamp(s, min = log_std_min, max = log_std_max)
        return m, s
    
    def sample(self, state):
        noise=1e-6
        m, s = self.forward(state) 
        std = s.exp()
        normal = Normal(m, std)
        
        
        ## Reparameterization (https://spinningup.openai.com/en/latest/algorithms/sac.html)
        # There are two sample functions in normal distributions one gives you normal sample ( .sample() ),
        # other one gives you a sample + some noise ( .rsample() )
        a = normal.rsample() # This is for the reparamitization
        tanh = torch.tanh(a)
        action = tanh * self.action_scale + self.action_bias
        
        logp = normal.log_prob(a)
        # Comes from the appendix C of the original paper for scaling of the action:
        logp =logp-torch.log(self.action_scale * (1 - tanh.pow(2)) + noise)
        logp = logp.sum(1, keepdim=True)
        
        
        if len(action)>1:
            print('stop')
        

        return action, logp


# Action-Value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()

        # Critic-1: Q1 
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Critic-2: Q2 
        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_) # OpReplayMemorytional


    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q1 = F.relu(self.linear1(state_action))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(state_action))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2
    
# Buffer
class ReplayMemory:
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, element):
        if len(self.memory) < self.memory_capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.memory_capacity

    def sample(self):
        return list(zip(*random.sample(self.memory, self.batch_size)))

    def __len__(self):
        return len(self.memory)


class Sac_agent:
    def __init__(self, state_size, action_size, hidden_dim, high, low, memory_capacity, batch_size,
                 gamma, tau,num_updates, policy_freq, alpha):
        
         # Actor Network 
        self.actor = Actor(state_size, action_size,hidden_dim, high, low).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        

        # Critic Network and Target Network
        self.critic = Critic(state_size, action_size, hidden_dim).to(device)   
        
        self.critic_target = Critic(state_size, action_size, hidden_dim).to(device)        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
        # copy weights
        self.hard_update(self.critic_target, self.critic)
        
        # Value network and Target Network
        self.value = ValueNetwork(state_size, hidden_dim).to(device)
        self.value_optim =optim.Adam(self.value.parameters(), lr=1e-4)
        self.target_value = ValueNetwork(state_size, hidden_dim).to(device)
        
        # copy weights
        self.hard_update(self.target_value, self.value)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = ReplayMemory(memory_capacity, batch_size)
        self.gamma = gamma
        self.tau = tau
        self.num_updates = num_updates
        self.iters = 0
        self.policy_freq=policy_freq
        
        ## For Dynamic Adjustment of the Parameter alpha (entropy coefficient) according to Gaussion policy (stochastic):
        self.target_entropy = -float(self.action_size) # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2, -11 for Reacher-v2)
        self.log_alpha = torch.zeros(1, requires_grad=True, device = device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4)
        self.alpha = alpha
        
        
    def hard_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data)
            
    def soft_update(self, target, network):
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
            
    def learn(self, batch):
        for _ in range(self.num_updates):                
            state, action, reward, next_state, mask = batch

            state = torch.tensor(state).to(device).float()
            next_state = torch.tensor(next_state).to(device).float()
            reward = torch.tensor(reward).to(device).float().unsqueeze(1)
            action = torch.tensor(action).to(device).float()
            mask = torch.tensor(mask).to(device).int().unsqueeze(1)
                         
            value_current=self.value(state)
            value_next=self.target_value(next_state)
            act_next, logp_next = self.actor.sample(next_state)
                
            ## Compute targets
            Q_target_main = reward + self.gamma*mask*value_next # Eq.8 of the original paper

            ## Update Value Network
            Q_target1, Q_target2 = self.critic_target(next_state, act_next) 
            min_Q = torch.min(Q_target1, Q_target2)
            value_difference = min_Q - logp_next # substract min Q value from the policy's log probability of slelecting that action
            value_loss = 0.5 * F.mse_loss(value_current, value_difference) # Eq.5 from the paper
            # Gradient steps 
            self.value_optim.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optim.step()
            
            ## Update Critic Network       
            critic_1, critic_2 = self.critic(state, action)
            critic_loss1 = 0.5*F.mse_loss(critic_1, Q_target_main) # Eq. 7 of the original paper
            critic_loss2 = 0.5* F.mse_loss(critic_2, Q_target_main) # Eq. 7 of the original paper
            total_critic_loss=critic_loss1+ critic_loss2  
            # Gradient steps
            self.critic_optimizer.zero_grad()
            total_critic_loss.backward() 
            self.critic_optimizer.step() 

            ## Update Actor Network with Entropy Regularized (look at the link for entropy regularization)
            act_pi, log_pi = self.actor.sample(state) # Reparameterize sampling
            Q1_pi, Q2_pi = self.critic(state, act_pi)
            min_Q_pi = torch.min(Q1_pi, Q2_pi)
            actor_loss =-(min_Q_pi-self.alpha*log_pi ).mean() # For minimization
            # Gradient steps
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            ## Dynamic adjustment of the Entropy Parameter alpha (look at the link for entropy regularization)
            alpha_loss = (-self.log_alpha * (log_pi.detach()) - self.log_alpha* self.target_entropy).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            
            ## Soft Update Target Networks using Polyak Averaging
            if (self.iters % self.policy_freq == 0):         
                self.soft_update(self.critic_target, self.critic)
                self.soft_update(self.target_value, self.value)
        
    def act(self, state):
        state =  torch.tensor(state).unsqueeze(0).to(device).float()
        action, logp = self.actor.sample(state)
        return action.cpu().data.numpy()[0]
    
    def step(self):
        self.learn(self.memory.sample())
        
    def save(self):
        torch.save(self.actor.state_dict(), "path_track_actor.pkl")
        torch.save(self.critic.state_dict(), "path_track_critic.pkl")
        

def sac(episodes):
    agent = Sac_agent(state_size = state_size, action_size = action_size, hidden_dim = hidden_dim, high = high, low = low, 
                  memory_capacity = memory_capacity, batch_size = batch_size, gamma = gamma, tau = tau, 
                  num_updates = num_updates, policy_freq =policy_freq, alpha = entropy_coef)
    time_start = time.time()
    reward_list = []
    avg_score_deque = deque(maxlen = 100)
    avg_scores_list = []
    mean_reward = -20000
    # Boundaries of the variable lookahead distance lf
    lb=2
    ub=5
    
    
    target_speed = 10.0  
    cx = np.arange(0, 50, 0.5)                                  # cx (list): A list of the x coordinates of the target course.
    cy = [math.sin(ix / 5.0) * ix / 5.0 for ix in cx]           # cy (list): A list of the y coordinates of the target course.
    
   
    
    for i in range(episodes):
        times = 0.0 
        state_control = InitialState_Update(x=cx[0], y=cy[0], yaw=0.0, v=0.0) 
        state_save=StateSave()
        state_save.append_all(times, state_control)  
        target_course = TargetCourse(cx, cy)  
        lastIndex = len(cx) - 1 
        
        lf=Lfc
        target_ind, _ = target_course.search_target_index(state_control,lf)   
        p= proportional_control(target_speed, state_control.v) 
        steering_angle, target_ind = pure_pursuit_steer_control(state_control, target_course, target_ind,lf)             
        CTE=cross_track_error(cx, cy, state_control.x, state_control.y)
        
        

        # initial state
        state_RL=[CTE, steering_angle, lf] # RL state: [Cross-track error, steering angle, lookahead distance]
        # action: lookahead distance
        total_reward = 0
        done = False
        episode_steps = 0
        while not done:
            episode_steps+=1 
            agent.iters=episode_steps
          
            
            if  episode_steps < 10: # To increase exploration, 10 orj
                population_size=1
                dim=1
                action=np.random.uniform(0,1,(population_size,dim)) *(ub-lb)+lb #action=lf (look ahead distance)
                lf=action[0][0]
                #action=action[0]
                lf= np.maximum(lb, lf)  # lower boundary check
                lf = np.minimum(ub, lf )  # Upper bound check
                action=[lf]
                #action = env.action_space.sample() # Orj

            else:
                action = agent.act(state_RL) # to sample the actions by Gaussian 
                lf=action[0]
                lf= np.maximum(lb, lf)  # lower boundary check
                lf = np.minimum(ub, lf )  # Upper bound check
                action=[lf]
            
            print(action)
            state_control.update(p, steering_angle)
            times += dt
            state_save.append_all(times, state_control)
            p= proportional_control(target_speed, state_control.v) 
            steering_angle, target_ind = pure_pursuit_steer_control(state_control, target_course, target_ind,lf)             
            CTE=cross_track_error(cx, cy, state_control.x, state_control.y) 
            print('CTE: ', CTE)
            state_RL_next=[CTE,steering_angle,lf]
            reward, done=reward_calculate(state_RL_next)
            #next_state, reward, done, _ = env.step(action) # orj
            
             # Ignore the "done" signal if it comes from hitting the time horizon.
            if episode_steps == max_episode_steps: # if the current episode has reached its maximum allowed steps
                mask = 1
            else:
                mask = float(not done)
            
            if (len(agent.memory) >= agent.memory.batch_size): 
                agent.step()
            
            if total_reward<-10000:
                 print('Dur')
            
            total_reward += reward
            state_RL=state_RL_next
            #state = next_state # orj
            print(f"episode: {i+1}, steps:{episode_steps}, current reward: {reward}")
            agent.memory.push((state_RL, action, reward, state_RL_next, mask))
            #env.render()
        
        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)
        
    plt.plot(reward_list)
    agent.save()
    print(f"episode: {i+1}, steps:{episode_steps}, current reward: {total_reward}, max reward: {np.max(reward_list)}")
    
                
    return reward_list, avg_scores_list


# action: lookahead distance
action_size = 1
print(f'dim of each action = {action_size}')
# state: cross-track error, steering angle, lookahead distance
state_size = 3
print(f'dim of state = {state_size}')
# boundaries of the action
high=5.0
low=2.0


max_episode_steps=1000 # if the agent does not reach "done" in "max_episode_steps", mask is 1
batch_size=20 # size that will be sampled from the replay memory that has maximum of "memory_capacity"
memory_capacity = 200 # 2000, maximum size of the memory
gamma = 0.99            
tau = 0.005               
num_of_train_episodes = 1500
num_updates = 1 # how many times you want to update the networks in each episode
policy_freq= 2 # lower value more probability to soft update,  policy frequency for soft update of the target network borrowed by TD3 algorithm
entropy_coef = 0.2 # For entropy regularization
num_of_test_episodes=200
hidden_dim=256

# Traning agent
#reward, avg_reward = sac(num_of_train_episodes)



# Testing
best_actor = Actor(state_size, action_size, hidden_dim = hidden_dim, high = high, low = low)
best_actor.load_state_dict(torch.load("path_track_actor.pkl"))        
best_actor.to(device) 
reward_test = []
best_action_list = []
best_reward_list=[]


target_speed = 10.0  
cx = np.arange(0, 50, 0.5)                                  # cx (list): A list of the x coordinates of the target course.
cy = [math.sin(ix / 5.0) * ix / 5.0 for ix in cx]  


for i in range(num_of_test_episodes):
    times = 0.0 
    state_control = InitialState_Update(x=cx[0], y=cy[0], yaw=0.0, v=0.0) 
    state_save=StateSave()
    state_save.append_all(times, state_control)  
    target_course = TargetCourse(cx, cy)  
    lastIndex = len(cx) - 1 
        
    lf=Lfc
    target_ind, _ = target_course.search_target_index(state_control,lf)   
    p= proportional_control(target_speed, state_control.v) 
    steering_angle, target_ind = pure_pursuit_steer_control(state_control, target_course, target_ind,lf)             
    CTE=cross_track_error(cx, cy, state_control.x, state_control.y)
        
    # initial state
    state_RL=[CTE, steering_angle, lf] 
    local_reward = 0
    done = False
    reward_previous=-1000
    episode_steps = 0
    while not done or  lastIndex > target_ind:
        episode_steps+=1
        print('test episonde:',i, 'test iteration: ', episode_steps)
        #state_RL =  torch.tensor(state_RL).to(device).float()
        state_RL =  torch.tensor(state_RL).unsqueeze(0).to(device).float()
        action,_=best_actor.sample(state_RL)
        mean,logp = best_actor(state_RL)        
        mean = mean.cpu().data.numpy() 
        state_control.update(p, steering_angle)
        state_save.append_all(times, state_control)
        times += dt
        p= proportional_control(target_speed, state_control.v) 
        steering_angle, target_ind = pure_pursuit_steer_control(state_control, target_course, target_ind,lf)             
        CTE=cross_track_error(cx, cy, state_control.x, state_control.y)
        lf=action[0]
        state_RL_next=[CTE,steering_angle,lf] 
        reward, done=reward_calculate(state_RL_next)
        if reward>reward_previous:
            best_lookahead=lf
            best_reward=reward
            reward_previous=reward
        #state, r, done, _ = new_env.step(action)
        local_reward += float(reward[0])
        state_RL=state_RL_next
        if episode_steps == max_episode_steps: # if the current episode has reached its maximum allowed steps
                done=True
        
                
    
    
    reward_test.append(local_reward)
    best_action_list.append(best_lookahead)
    best_reward_list.append(best_reward)

optimal_lookahead=best_action_list[best_reward_list.index(max(best_reward_list))]




import plotly.graph_objects as go
x = np.array(range(len(reward_test)))
m = np.mean(reward_test)


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=reward_test, name='test reward',
                                 line=dict(color="green", width=1)))

fig.add_trace(go.Scatter(x=x, y=[m]*len(reward_test), name='average reward',
                                 line=dict(color="red", width=1)))
    
fig.update_layout(title="SAC",
                           xaxis_title= "test",
                           yaxis_title= "reward")
fig.show()

print("average reward:", m)


