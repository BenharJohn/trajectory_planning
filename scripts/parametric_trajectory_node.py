#!/usr/bin/env python

# Import necessary libraries
import rospy  # ROS Python client library
import numpy as np  # For mathematical operations
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # Message types for publishing trajectory
from sensor_msgs.msg import JointState  # Message type for subscribing to joint states
import trac_ik_python.trac_ik as tracik  # TRAC-IK library for inverse kinematics
from tf.transformations import quaternion_from_euler  # For quaternion conversion
from scipy.interpolate import CubicSpline  # To interpolate joint trajectory with cubic splines

class ParametricTrajectoryPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('parametric_trajectory_publisher')

        # Parameters for trajectory generation
        self.radius = rospy.get_param('~radius', 0.2)  # Radius of the circular trajectory in meters
        self.angular_velocity = rospy.get_param('~angular_velocity', 0.5)  # Angular velocity in radians per second
        self.center = rospy.get_param('~center', [0.5, 0.0, 0.5])  # Center of the circle [x, y, z]
        self.duration = rospy.get_param('~duration', 20.0)  # Total duration of the trajectory in seconds
        self.dt = rospy.get_param('~dt', 0.05)  # Time step for trajectory generation in seconds

        # Initialize TRAC-IK solver for inverse kinematics
        self.base_link = rospy.get_param('~base_link', 'base_link')  # Base link of the UR5 robot
        self.tip_link = rospy.get_param('~tip_link', 'wrist_3_link')  # End-effector link of the UR5 robot
        self.ik_solver = tracik.IK(self.base_link, self.tip_link, timeout=0.1)  # IK solver with a timeout of 0.1 seconds

        # Retrieve joint names from the IK solver
        ik_joint_names = self.ik_solver.joint_names
        rospy.loginfo("Joint names from IK solver: {}".format(ik_joint_names))

        # Define expected joint names from the robot controller
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Map indices between IK solver joint names and controller joint names
        self.controller_to_ik_indices = [ik_joint_names.index(name) for name in self.joint_names]
        self.ik_to_controller_indices = [self.joint_names.index(name) for name in ik_joint_names]

        # Publisher to send joint trajectory commands
        self.traj_pub = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)

        # Subscriber to receive the current joint states
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.current_joint_state = None  # To store the latest joint states

        # Wait for the first joint state message
        while self.current_joint_state is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Generate and publish the trajectory
        self.publish_trajectory()

    def joint_state_callback(self, msg):
        """Callback function to receive joint states."""
        self.current_joint_state = msg

    def compute_joint_positions(self, x, y, z, q_seed):
        """
        Use TRAC-IK to compute joint positions for a desired Cartesian position.
        :param x, y, z: Desired end-effector position in Cartesian space.
        :param q_seed: Initial seed for solving IK.
        :return: Joint positions reordered for the controller, and the seed for the next computation.
        """
        try:
            # Desired position and orientation
            pos = [x, y, z]
            rot_quat = quaternion_from_euler(0, np.pi, 0)  # Fixed orientation with end-effector pointing down
            rot = [rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]]

            # Solve IK to find joint angles
            joint_positions = self.ik_solver.get_ik(
                q_seed, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]
            )

            if joint_positions is not None:
                # Reorder joint positions for the robot controller
                joint_positions_ordered = [joint_positions[self.controller_to_ik_indices[idx]] for idx in range(len(self.joint_names))]
                return joint_positions_ordered, joint_positions
            else:
                return None, q_seed  # Return the previous seed if no solution is found
        except Exception as e:
            rospy.logerr("Error computing IK: {}".format(e))
            return None, q_seed

    def publish_trajectory(self):
        """Generate and publish a parametric circular trajectory."""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Time steps for the trajectory
        time_steps = np.arange(0, self.duration + self.dt, self.dt)

        # Store waypoints and times for interpolation
        joint_waypoints = []
        times = []

        # Get the initial joint positions from the robot state
        q_init_controller = self.current_joint_state.position
        q_init = [q_init_controller[self.ik_to_controller_indices[idx]] for idx in range(len(self.joint_names))]

        # Compute joint positions for each time step
        for t in time_steps:
            theta = self.angular_velocity * t  # Compute the angular position on the circle
            x = self.center[0] + self.radius * np.cos(theta)  # X-coordinate of the point
            y = self.center[1] + self.radius * np.sin(theta)  # Y-coordinate of the point
            z = self.center[2]  # Z-coordinate (constant)

            joint_positions_ordered, q_init = self.compute_joint_positions(x, y, z, q_init)

            if joint_positions_ordered is not None:
                joint_waypoints.append(joint_positions_ordered)
                times.append(t)
            else:
                rospy.logwarn("No IK solution found at time {:.2f}s".format(t))

        # Check if we have enough waypoints for a valid trajectory
        if len(joint_waypoints) < 2:
            rospy.logerr("Not enough valid joint waypoints to create a trajectory.")
            return

        # Interpolate the trajectory using cubic splines
        trajectory_points = self.interpolate_joint_trajectory(joint_waypoints, times)

        # Assign the interpolated points to the trajectory message
        trajectory_msg.points = trajectory_points

        # Publish the trajectory if valid points are available
        if trajectory_points:
            self.traj_pub.publish(trajectory_msg)
            rospy.loginfo("Parametric trajectory published with {} points.".format(len(trajectory_points)))
            rospy.sleep(self.duration + 2)  # Keep node alive to allow trajectory execution
        else:
            rospy.logerr("No valid trajectory points generated after interpolation.")

    def interpolate_joint_trajectory(self, joint_waypoints, times):
        """
        Interpolates joint positions using cubic splines for smooth motion.
        :param joint_waypoints: List of joint positions at each waypoint.
        :param times: List of time stamps for each waypoint.
        :return: List of interpolated JointTrajectoryPoints.
        """
        joint_waypoints = np.array(joint_waypoints)
        times = np.array(times)

        num_joints = joint_waypoints.shape[1]
        time_samples = np.arange(0, times[-1] + self.dt, self.dt)

        trajectory_points = []

        # Fit a cubic spline for each joint
        splines = [CubicSpline(times, joint_waypoints[:, j], bc_type='clamped') for j in range(num_joints)]

        for t in time_samples:
            positions = [spline(t) for spline in splines]
            velocities = [spline.derivative()(t) for spline in splines]
            accelerations = [spline.derivative(2)(t) for spline in splines]

            # Create a trajectory point with interpolated values
            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            point.accelerations = accelerations
            point.time_from_start = rospy.Duration.from_sec(t)
            trajectory_points.append(point)

        return trajectory_points

if __name__ == '__main__':
    try:
        ParametricTrajectoryPublisher()
    except rospy.ROSInterruptException:
        pass
