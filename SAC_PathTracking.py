"""

Path tracking simulation with pure pursuit steering and PID speed control.

References :

    1)    Experimental Validation of a Kinematic Bicycle Model Predictice Control With Lateral Acceleration Consideration : IFAC PapersOnLine 52-8 (2019) 289–294 
    2)    Automatic Steering Methods for Autonomous Automobile Path Tracking : CMU-RI-TR-09-08  
    3)    D. S. Lal, A. Vivek and G. Selvaraj, "Lateral control of an autonomous vehicle based on Pure Pursuit algorithm," 2017 International Conference on Technological Advancements in Power and Energy ( TAP Energy), Kollam, India, 2017, pp. 1-8, doi: 10.1109/TAPENERGY.2017.8397361.  
    4)    R. Wang, Y. Li, J. Fan, T. Wang and X. Chen, "A Novel Pure Pursuit Algorithm for Autonomous Vehicles Based on Salp Swarm Algorithm and Velocity Controller," in IEEE Access, vol. 8, pp. 166525-166540, 2020, doi: 10.1109/ACCESS.2020.3023071.


"""
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import timeit



k = 0.1     # k: This is the look forward gain, which scales the look-ahead distance based on the vehicle's velocity.                                   [Pure Pursuit]
Lfc = 2.5   # Lfc: The look-ahead distance, which determines how far ahead the vehicle should consider when choosing the target point on the path.      [Pure Pursuit]
WB = 0.1    # WB: The wheelbase of the vehicle, which is the distance between the front and rear axles.                                                 [Pure Pursuit]

Kp = 1.0    # Kp: The speed proportional gain, used in the proportional control law to adjust the acceleration.                                         [PID Speed Control]
Ki = 0.1    # Ki: The speed integral gain, used in the integral control law to adjust the acceleration.                                                 [PID Speed Control]
Kd = 0.01   # Kd: The speed derivative gain, used in the derivative control law to adjust the acceleration.                                             [PID Speed Control]
dt = 0.1    # dt: The time step used in the simulation, representing the interval at which updates are calculated. # 0.1                                    [PID Speed Control]




def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v),(1/beta))
    step = np.divide(u,zz)
    return step


def reward(cx, cy, x_car, y_car):

  cxy_car=[cx, cy]
  xy_car=[x_car, y_car]
  cxy_car = np.array(cxy_car).T

  cte_list = [np.sqrt((x - xy_car[0])**2 + (y - xy_car[1])**2) for x, y in cxy_car]
  CTE = min(cte_list)
 

  fitness_rmse= CTE / len(cte_list) 
  

  return fitness_rmse



show_animation = True


class State:

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
                                                                    # Y = sin(ϴ) [ref:3. pp:3. eq:9]

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x                                  # dx (float): The distance between the rear axle and the given point along the x-axis.
        dy = self.rear_y - point_y                                  # dy (float): The distance between the rear axle and the given point along the y-axis.
        return math.hypot(dx, dy)                                   # Returns the distance between the rear axle and the given point using the Pythagorean theorem. (c = sqrt(a^2 + b^2)


class States:

    def __init__(self ):                                            # This class is used to store the vehicle's state at each time step.
        self.x = []                                                 # x (list): A list of the vehicle's x position at each time step.
        self.y = []                                                 # y (list): A list of the vehicle's y position at each time step.
        self.yaw = []                                               # yaw (list): A list of the vehicle's orientation at each time step.
        self.v = []                                                 # v (list): A list of the vehicle's velocity at each time step.
        self.t = []                                                 # t (list): A list of the time at each time step.

    def append(self, t, state):                                     # This method is used to append the vehicle's state at a given time step to the corresponding lists.
        self.x.append(state.x)                                      # t (float): The time at the given time step.
        self.y.append(state.y)                                      # state (State): The vehicle's state at the given time step.
        self.yaw.append(state.yaw)                                  # This method is used to append the vehicle's state at a given time step to the corresponding lists.
        self.v.append(state.v)                                      # This method is used to append the vehicle's state at a given time step to the corresponding lists.
        self.t.append(t)                                            # This method is used to append the vehicle's state at a given time step to the corresponding lists.


def proportional_control(target, current):                          # This function calculates the acceleration required to adjust the vehicle's velocity to the target velocity.
    a = Kp * (target - current)                                     # target (float): The target velocity.

    return a                                                        # current (float): The current velocity.


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
        """ 
        The method returns the index of the target point (ind) and the calculated look-ahead distance (Lf) to be used in other parts of the simulation, specifically in the pure pursuit steering algorithm.

        """
    """ 
        Overall, the aim of this method is to determine the index of the target point on the trajectory that the vehicle should aim to follow. 
        It accounts for the vehicle's current position, orientation, and desired look-ahead distance to ensure accurate tracking of the trajectory. 
        The method balances efficiency (by using the previously found nearest point) and accuracy (by iteratively searching for the next appropriate point within the look-ahead distance).
    """

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


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
        Function Parameters:
            x and y: The coordinates of the arrow's starting point (vehicle's position).
            yaw: The orientation angle of the arrow (vehicle's orientation).
            length: The length of the arrow representing the vehicle.
            width: The width of the arrow.
            fc (facecolor): The fill color of the arrow.
            ec (edgecolor): The edge color of the arrow.

    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)

            """ 
                Arrow Plotting for Single Point:
                    This condition checks if the provided x coordinate is a single value (i.e., not a list or array). If it's not a single value, it indicates that multiple points are being plotted.
                    If the coordinates are in lists or arrays (multiple points), the function recursively calls itself for each point in the lists (x, y, yaw) to create individual arrow plots.
            """
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

        """ 
            Arrow Plotting for Multiple Points:
                If the provided x coordinate is a single value, this part of the function is executed to plot a single arrow.
                plt.arrow is used to create the arrow. It takes the starting point (x, y), the change in x (length * math.cos(yaw)) and y (length * math.sin(yaw)) coordinates based on the arrow's orientation.
                The fc and ec parameters define the fill and edge colors of the arrow.
                head_width and head_length determine the size of the arrowhead.
                plt.plot(x, y) is used to mark the starting point of the arrow with a point on the plot.
        """
    """
        In summary, the aim of this function is to visualize the vehicle's position and orientation on a matplotlib plot.
        It achieves this by creating arrow representations that depict the vehicle's movement and direction.
        It's a crucial part of the simulation for visually tracking the vehicle's path and orientation.

    """



def main():
    
    cx = np.arange(0, 50, 0.5)                                  # cx (list): A list of the x coordinates of the target course.
    cy = [math.sin(ix / 5.0) * ix / 5.0 for ix in cx]           # cy (list): A list of the y coordinates of the target course.
    
    #target_speed = 10.0 / 3.6                                   # target_speed (float): The target speed of the vehicle in m/s.
    target_speed = 10.0 
    T = 100   # 100                                                # T (float): The total simulation time in seconds.
    
    
    plt.plot(cx, cy, ".r", label="Target trajectory")
    
    plt.legend()
    #plt.xlabel("x[m]")
    #plt.ylabel("y[m]")
    plt.axis("equal")
    plt.xlim((0,45))
    plt.grid(True)
    plt.show()
    
    
    dim=1
    SearchAgents_no=10
    lb=2
    ub=5
    Max_iter=T
     # initialize the location and Energy of the rabbit
    Rabbit_Location=np.zeros(dim)
    Rabbit_Energy=float("inf")  #change this to -inf for maximization problems
    #Initialize the locations of Harris' hawks
    X=np.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
    
    #Initialize convergence
    #convergence_curve=np.zeros(Max_iter)
    convergence_curve=[]
    
    #Initialize the locations of Harris' hawks
    X=np.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb
    
   
    
    
    
    state = State(x=cx[0], y=cy[0], yaw=0.0, v=0.0)               # state (State): The vehicle's initial state. The vehicle is initialized at the origin with zero velocity and orientation.

    lastIndex = len(cx) - 1                                     # lastIndex (int): The index of the last point on the target course.
    time = 0.0                                                  # time (float): The current time in the simulation.
    states = States()                                           # states (States): The vehicle's states at each time step.
    states.append(time, state)                                  # This line appends the initial state to the states object.
    target_course = TargetCourse(cx, cy)                        # target_course (TargetCourse): The target course that the vehicle should follow.
    lf=2.5 # initial lookahead distance
    target_ind, _ = target_course.search_target_index(state,lf)    # target_ind (int): The index of the target point on the target course that the vehicle should follow.
    t=0 # iteration
    start = timeit.default_timer()
    dt = 0.1 
    while T >= time and lastIndex > target_ind:                 # This loop continues as long as the simulation time (T) has not exceeded and the target index is within the valid range.
        
        # Main-Update
        ai = proportional_control(target_speed, state.v)        # ai (float): The acceleration required to adjust the vehicle's velocity to the target velocity.
        di, target_ind = pure_pursuit_steer_control(            # di (float): The steering angle required to follow the target point on the trajectory.
            state, target_course, target_ind,lf)                   # target_ind (int): The index of the target point on the target course that the vehicle should follow.                                   
        state.update(ai, di) 
        time += dt
        states.append(time, state)
         
        print("state-x:", state.x)
        
        if t==0: # initialization
            #fitness=reward(cx, cy, states.x, states.y)
            for i in range(0,SearchAgents_no):
            
                # Check boundries         
                X[i,:]=np.clip(X[i,:], lb, ub)
                lf=X[i,:]
                di, target_ind = pure_pursuit_steer_control(            # di (float): The steering angle required to follow the target point on the trajectory.
                state, target_course, target_ind,lf)                   # target_ind (int): The index of the target point on the target course that the vehicle should follow.                                   
                state.update(ai, di) 
                
                fitness=reward(cx, cy, state.x, state.y) # cx, cy --> target trajectory
            
                # fitness of locations
                #fitness=objf(X[i,:])
            
                # Update the location of Rabbit
                if fitness<Rabbit_Energy: # Change this to > for maximization problem
                    Rabbit_Energy=fitness 
                    Rabbit_Location=X[i,:].copy()
                    X_best= Rabbit_Location
                    Fitness_best=Rabbit_Energy
            
        E1=2*(1-(t/T)) # factor to show the decreaing energy of rabbit    
        
        # Update the location of Harris' hawks 
        for i in range(0,SearchAgents_no):

            E0=2*random.random()-1;  # -1<E0<1
            Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy)>=1:
                #Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no*random.random())
                X_rand = X[rand_Hawk_index, :]
                if q<0.5:
                    # perch based on other family members
                    X[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*X[i,:])

                elif q>=0.5:
                    #perch on a random tall tree (random site inside group's home range)
                    X[i,:]=(Rabbit_Location - X.mean(0))-random.random()*((ub-lb)*random.random()+lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy)<1:
                #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                #phase 1: ----- surprise pounce (seven kills) ----------
                #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r=random.random() # probablity of each event
                
                if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                    X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength=2*(1- random.random()); # random jump strength of the rabbit
                    X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                
                #phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                    #rabbit try to escape by many zigzag deceptive motions
                    Jump_strength=2*(1-random.random())
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:]);
                    # Check boundries         
                    X1=np.clip(X1, lb, ub)
                    lf=X1
                    di, target_ind = pure_pursuit_steer_control(            # di (float): The steering angle required to follow the target point on the trajectory.
                    state, target_course, target_ind,lf)                   # target_ind (int): The index of the target point on the target course that the vehicle should follow.                                   
                    state.update(ai, di) 
                    fitness_X1=reward(cx, cy, state.x, state.y) 


                    if fitness_X1< fitness: # improved move?
                        X[i,:] = X1.copy()
                        X_best= X[i,:]
                        Fitness_best=fitness_X1
                    else: # hawks perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+np.multiply(np.random.randn(dim),Levy(dim))
                        # Check boundries         
                        X2=np.clip(X2, lb, ub)
                        lf=X2
                        di, target_ind = pure_pursuit_steer_control(            # di (float): The steering angle required to follow the target point on the trajectory.
                        state, target_course, target_ind,lf)                   # target_ind (int): The index of the target point on the target course that the vehicle should follow.                                   
                        state.update(ai, di) 
                        fitness_X2=reward(cx, cy, state.x, state.y) 
                        
                        if fitness_X2< fitness:
                            X[i,:] = X2.copy()
                if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                     Jump_strength=2*(1-random.random())
                     X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                     # Check boundries         
                     X1=np.clip(X1, lb, ub)
                     lf=X1
                     di, target_ind = pure_pursuit_steer_control(            # di (float): The steering angle required to follow the target point on the trajectory.
                     state, target_course, target_ind,lf)                   # target_ind (int): The index of the target point on the target course that the vehicle should follow.                                   
                     state.update(ai, di) 
                     fitness_X1=reward(cx, cy, state.x, state.y) 
                     if fitness_X1< fitness: # improved move?
                        X[i,:] = X1.copy()
                        X_best= X[i,:]
                        Fitness_best=fitness_X1
                     else: # Perform levy-based short rapid dives around the rabbit
                         X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+np.multiply(np.random.randn(dim),Levy(dim))
                         # Check boundries         
                         X2=np.clip(X2, lb, ub)
                         lf=X2
                         di, target_ind = pure_pursuit_steer_control(            # di (float): The steering angle required to follow the target point on the trajectory.
                         state, target_course, target_ind,lf)                   # target_ind (int): The index of the target point on the target course that the vehicle should follow.                                   
                         state.update(ai, di) 
                         fitness_X2=reward(cx, cy, state.x, state.y) 
                         
                         if fitness_X2< fitness:
                            X[i,:] = X2.copy()
                            X_best= X[i,:]
                            Fitness_best=fitness_X2
                
        #convergence_curve[t]=Rabbit_Energy
        convergence_curve.append(Fitness_best)
        #if (t%1==0):
               #print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
        t+=1
        lf=X_best
        #s.best =Rabbit_Energy 
        #s.bestIndividual = Rabbit_Location

        """ 
            This section calculates the acceleration and steering angle required to follow the target point on the trajectory.
            The vehicle's state is updated using the calculated control inputs.
            The simulation time and vehicle states are updated.
            If show_animation is enabled, the vehicle's trajectory, target point, and animation are plotted for visualization.
        """
        
        '''
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)
        '''
        
    
        

    
    stop = timeit.default_timer()
    estimated_time=stop - start
    # Test
    assert lastIndex >= target_ind, "Cannot goal"
    #states.y[0]=0
    plt.plot(cx, cy, ".r", label="Target trajectory")
    plt.plot(states.x, states.y, "-b", label="Found trajectory")
    plt.legend()
    #plt.xlabel("x[m]")
    #plt.ylabel("y[m]")
    #plt.axis("equal")
    plt.grid(True)
    plt.xlim((0,45))
    plt.show()
    print('stop')


    '''
    if show_animation:  
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()
    '''

if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()

    """ 
        The aim of this section is to orchestrate the entire simulation process.
        It initializes the vehicle's state, calculates control inputs, updates the vehicle's state, and plots the results for visualization.
        This demonstrates how the pure pursuit path tracking algorithm and PID speed control work together to guide the vehicle along the desired trajectory.
    """