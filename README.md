# UR5 Trajectory Planning  

This repository contains Python scripts for planning and executing trajectories for the UR5 robotic arm. The focus is on generating smooth and continuous motions using advanced trajectory planning techniques.  

## Repository Contents  

1. **`parametric_trajectory_node.py`:**  
   - Generates a circular trajectory for the UR5 end-effector.  
   - Uses the Python library `scipy` to perform cubic spline interpolation for smooth transitions.  
   - Implements inverse kinematics with the TRAC-IK solver to calculate joint positions for the desired Cartesian poses.  

2. **`ur5_polynomial_trajectory.py`:**  
   - Implements a multi-segment trajectory with smooth transitions between waypoints.  
   - Uses quintic polynomial coefficients to ensure continuity of position, velocity, and acceleration.  

## Features  

- **Smooth Trajectory Planning:**  
  - Circular trajectories with parametric equations.  
  - Polynomial trajectories for multi-waypoint paths.  

- **Inverse Kinematics:**  
  - TRAC-IK integration to compute joint positions from Cartesian trajectories.  

- **Customization:**  
  - Easily modify trajectory parameters such as radius, angular velocity, waypoint times, and duration.  

## Prerequisites  

- ROS (Robot Operating System)  
- TRAC-IK library  
- Python libraries: `rospy`, `numpy`, `scipy`  

## Usage  

1. Clone the repository:  
   ```bash
   git clone https://github.com/BenharJohn/trajectory_planning.git
   cd trajectory_planning
   ```  

2. Ensure the scripts are executable:  
   ```bash
   chmod +x scripts/*.py
   ```  

3. Run the desired script:  
   ```bash
   rosrun ur5_trajectory_control parametric_trajectory_node.py
   ```  

   OR  

   ```bash
   rosrun ur5_trajectory_control ur5_polynomial_trajectory.py
   ```  

## Notes  

- The initial motion of the robot from its starting position to the circle in `parametric_trajectory_node.py` has not been explicitly slowed down. You can modify this behavior if needed.  
- Adjust the parameters like waypoints, time steps, radius, and velocities directly in the scripts or as ROS parameters.  

## Future Additions  

- Dynamic adjustments for velocity and acceleration.  
- Enhanced obstacle avoidance during trajectory execution.  

Feel free to contribute or raise issues for any improvements or bug fixes.  

---  
**Author:** [Your Name]  
