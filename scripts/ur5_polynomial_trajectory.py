#!/usr/bin/env python

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Function to compute quintic polynomial coefficients for smoother trajectories
def compute_quintic_coefficients(q0, qf, v0, vf, T):
    a0 = q0
    a1 = v0
    a2 = 0
    a3 = (20 * (qf - q0) - (8 * vf + 12 * v0) * T) / (2 * T ** 3)
    a4 = (30 * (q0 - qf) + (14 * vf + 16 * v0) * T) / (2 * T ** 4)
    a5 = (12 * (qf - q0) - (6 * vf + 6 * v0) * T) / (2 * T ** 5)
    return a0, a1, a2, a3, a4, a5

# Function to generate a multi-segment trajectory with smooth transitions across waypoints
def generate_continuous_trajectory(waypoints, waypoint_times, dt):
    trajectory = []
    num_joints = len(waypoints[0])
    cumulative_time = 0  # To ensure strictly increasing time_from_start

    # Set initial velocities for the first waypoint
    v0 = [0.0] * num_joints  # Initial velocity at the start
    for i in range(len(waypoints) - 1):
        q0 = waypoints[i]
        qf = waypoints[i + 1]
        T = waypoint_times[i + 1] - waypoint_times[i]
        time_steps = np.arange(0, T + dt, dt)

        # For the last waypoint, set final velocity to zero
        if i == len(waypoints) - 2:
            vf = [0.0] * num_joints
        else:
            vf = None

        # Calculate quintic coefficients for each joint between waypoints
        coefficients = []
        for j in range(num_joints):
            if vf:
                coeff = compute_quintic_coefficients(q0[j], qf[j], v0[j], vf[j], T)
                v0[j] = vf[j]  # Update initial velocity for next segment
            else:
                # Estimate velocity for smooth transition
                v_est = (qf[j] - q0[j]) / T
                coeff = compute_quintic_coefficients(q0[j], qf[j], v0[j], v_est, T)
                v0[j] = v_est  # Update initial velocity for next segment
            coefficients.append(coeff)

        # Generate trajectory points
        for t in time_steps:
            positions, velocities, accelerations = [], [], []
            for j in range(num_joints):
                a0, a1, a2, a3, a4, a5 = coefficients[j]
                # Position
                q = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
                positions.append(q)
                # Velocity
                qd = a1 + 2 * a2 * t + 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
                velocities.append(qd)
                # Acceleration
                qdd = 2 * a2 + 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
                accelerations.append(qdd)

            point = JointTrajectoryPoint()
            point.positions = positions
            point.velocities = velocities
            point.accelerations = accelerations
            cumulative_time += dt  # Accumulate time for each point
            point.time_from_start = rospy.Duration.from_sec(cumulative_time)
            trajectory.append(point)

    return trajectory

def main():
    rospy.init_node('ur5_continuous_trajectory')

    # Publisher to the joint trajectory controller
    traj_pub = rospy.Publisher('/eff_joint_traj_controller/command', JointTrajectory, queue_size=10)

    # Joint names for UR5
    joint_names = [
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]

    # Wait for the publisher to be ready
    rospy.sleep(1)

    # Define waypoints
    waypoints = [
        np.array([0, -1.2, 1.5, 0, 0, 0]),                # Start position
        np.array([0.3, -1.0, 1.3, 0.4, 0.4, 0]),          # Mid position
        np.array([-0.3, -0.8, 1.0, 0.2, -0.3, 0.1]),      # End position
        np.array([0, -1.2, 1.5, 0, 0, 0])                 # Return to Start
    ]
    waypoint_times = [0, 5, 10, 15]  # Adjusted times

    # Trajectory parameters
    dt = 0.05  # Time step in seconds

    # Generate the continuous trajectory using quintic polynomials
    trajectory_points = generate_continuous_trajectory(waypoints, waypoint_times, dt)

    # Create the trajectory message
    trajectory_msg = JointTrajectory()
    trajectory_msg.joint_names = joint_names
    trajectory_msg.points = trajectory_points

    # Publish the trajectory
    traj_pub.publish(trajectory_msg)
    rospy.loginfo("Continuous trajectory published.")

    rospy.spin()

if __name__ == '__main__':
    main()
