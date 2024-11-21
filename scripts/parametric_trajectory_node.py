#!/usr/bin/env python

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import trac_ik_python.trac_ik as tracik
from tf.transformations import quaternion_from_euler
from scipy.interpolate import CubicSpline

class ParametricTrajectoryPublisher:
    def __init__(self):
        rospy.init_node('parametric_trajectory_publisher')

        # Parameters
        self.radius = rospy.get_param('~radius', 0.2)  # Circle radius in meters
        self.angular_velocity = rospy.get_param('~angular_velocity', 0.5)  # Rad/s
        self.center = rospy.get_param('~center', [0.5, 0.0, 0.5])  # Center of circle
        self.duration = rospy.get_param('~duration', 20.0)  # Total time in seconds
        self.dt = rospy.get_param('~dt', 0.05)  # Time step in seconds

        # Initialize TRAC-IK solver
        self.base_link = rospy.get_param('~base_link', 'base_link')
        self.tip_link = rospy.get_param('~tip_link', 'wrist_3_link')  # Adjust as per UR5 setup
        self.ik_solver = tracik.IK(self.base_link, self.tip_link, timeout=0.1)

        # Get joint names from the IK solver
        ik_joint_names = self.ik_solver.joint_names
        rospy.loginfo("Joint names from IK solver: {}".format(ik_joint_names))

        # Expected joint names from the controller
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # Create mappings between IK joint names and controller joint names
        self.controller_to_ik_indices = [ik_joint_names.index(name) for name in self.joint_names]
        self.ik_to_controller_indices = [self.joint_names.index(name) for name in ik_joint_names]

        # Publisher
        self.traj_pub = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)

        # Subscriber
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.current_joint_state = None

        # Wait for joint states
        while self.current_joint_state is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # Generate and publish trajectory
        self.publish_trajectory()

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def compute_joint_positions(self, x, y, z, q_seed):
        # Use TRAC-IK to compute joint positions for the desired pose
        try:
            # Define the pose in terms of position and a fixed orientation
            pos = [x, y, z]
            # Use an orientation that is feasible (e.g., end-effector pointing downwards)
            rot_quat = quaternion_from_euler(0, np.pi, 0)  # [roll, pitch, yaw]
            rot = [rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3]]

            # Solve IK using TRAC-IK
            joint_positions = self.ik_solver.get_ik(
                q_seed,  # Seed state in IK joint order
                pos[0], pos[1], pos[2],
                rot[0], rot[1], rot[2], rot[3]
            )

            if joint_positions is not None:
                # Reorder joint_positions from IK order to controller order
                joint_positions_ordered = [joint_positions[self.controller_to_ik_indices[idx]] for idx in range(len(self.joint_names))]
                return joint_positions_ordered, joint_positions  # Return both ordered for controller and original for seed
            else:
                return None, q_seed  # Return previous seed if no solution found
        except Exception as e:
            rospy.logerr("Error computing IK: {}".format(e))
            return None, q_seed

    def publish_trajectory(self):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        time_steps = np.arange(0, self.duration + self.dt, self.dt)
        cumulative_time = 0

        # Store joint positions at waypoints
        joint_waypoints = []
        times = []

        # Initial joint positions in controller order
        q_init_controller = self.current_joint_state.position

        # Map q_init to IK order
        q_init = [q_init_controller[self.ik_to_controller_indices[idx]] for idx in range(len(self.joint_names))]

        for t in time_steps:
            theta = self.angular_velocity * t
            x = self.center[0] + self.radius * np.cos(theta)
            y = self.center[1] + self.radius * np.sin(theta)
            z = self.center[2]

            joint_positions_ordered, q_init = self.compute_joint_positions(x, y, z, q_init)

            if joint_positions_ordered is not None:
                joint_waypoints.append(joint_positions_ordered)
                times.append(t)
            else:
                rospy.logwarn("No IK solution found at time {:.2f}s".format(t))
                # Optionally, handle the absence of IK solutions

        # Check if we have enough waypoints
        if len(joint_waypoints) < 2:
            rospy.logerr("Not enough valid joint waypoints to create a trajectory.")
            return

        # Interpolate joint positions using cubic splines
        trajectory_points = self.interpolate_joint_trajectory(joint_waypoints, times)

        # Set the trajectory points
        trajectory_msg.points = trajectory_points

        if trajectory_points:
            # Publish the trajectory
            self.traj_pub.publish(trajectory_msg)
            rospy.loginfo("Parametric trajectory published with {} points.".format(len(trajectory_points)))

            # Keep the node alive until the trajectory is executed
            rospy.sleep(self.duration + 2)
        else:
            rospy.logerr("No valid trajectory points generated after interpolation.")

    def interpolate_joint_trajectory(self, joint_waypoints, times):
        joint_waypoints = np.array(joint_waypoints)
        times = np.array(times)

        num_joints = joint_waypoints.shape[1]

        # Create time samples for the entire duration
        total_time = times[-1]
        time_samples = np.arange(0, total_time + self.dt, self.dt)

        trajectory_points = []

        # For each joint, fit a cubic spline
        splines = []
        for j in range(num_joints):
            # For each joint, get the positions at waypoints
            q = joint_waypoints[:, j]
            # Fit a cubic spline
            spline = CubicSpline(times, q, bc_type='clamped')
            splines.append(spline)

        # Generate trajectory points at time_samples
        for t in time_samples:
            positions = []
            velocities = []
            accelerations = []
            for j in range(num_joints):
                spline = splines[j]
                q = spline(t)
                qd = spline.derivative()(t)
                qdd = spline.derivative(2)(t)
                positions.append(q)
                velocities.append(qd)
                accelerations.append(qdd)
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
