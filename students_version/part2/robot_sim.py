from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.geometry import HalfSpace
from pydrake.math import RotationMatrix, RigidTransform
from pydrake.common.eigen_geometry import Quaternion

import pinocchio

import numpy as np
import time

class RobotSim:
    def __init__(self, urdf_file, q_0, dt, robo_robot=None):
        self.dt = dt
        self.robo_robot = robo_robot
        self.builder = DiagramBuilder()
        self.robot, self.scene_graph = AddMultibodyPlantSceneGraph(builder=self.builder, time_step=dt)
        Parser(self.robot, self.scene_graph).AddModels(urdf_file)
        self.q_0 = q_0
        self.AddGround(self.robot)
        self.robot.Finalize()
        self.robot.SetDefaultPositions(self.q_0)

        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()

        self.simulator = Simulator(self.diagram, self.diagram_context)
        self.simulator.Initialize()
        # self.simulator.set_publish_every_time_step(False)
        # self.simulator.set_target_realtime_rate(1.0)
        self.plant_context = self.robot.GetMyContextFromRoot(self.simulator.get_mutable_context())

        self.count = 0
        self._viz_update = max(1, int(0.02 / self.dt)) # Update visualization only every 0.02s

    def AddGround(self, plant):
        transparent_color = np.array([0.5, 0.5, 0.5, 0])
        p_GroundOrigin = [0, 0.0, 0.0]
        R_GroundOrigin = RotationMatrix.MakeXRotation(0.0)
        X_GroundOrigin = RigidTransform(R_GroundOrigin, p_GroundOrigin)
        surface_friction = CoulombFriction(static_friction=0.7, dynamic_friction=0.5)
        plant.RegisterCollisionGeometry(
            plant.world_body(), X_GroundOrigin, HalfSpace(), "ground_collision", surface_friction)
        plant.RegisterVisualGeometry(
            plant.world_body(), X_GroundOrigin, HalfSpace(), "ground_visual", transparent_color)

    def getJointNames(self):
        return self.robot.GetActuatorNames()

    def step(self):
        self.simulator.AdvanceTo(self.dt * self.count)
        self.count += 1
        if self.count % self._viz_update == 0: # update only every 0.02s
            self._update_visualization()

    # def step(self): 
    # 	t0 = time.perf_counter()
    # 	self.simulator.AdvanceTo(self.dt * self.count)  
    # 	t1 = time.perf_counter()
    # 	self.count += 1
    # 	if self.count % self._viz_update == 0: # update only every 0.1s
    #     	self._update_visualization()
    # 	t2 = time.perf_counter()
    
    # 	sim_ms   = (t1 - t0) * 1000 
    # 	viz_ms   = (t2 - t1) * 1000
    # 	total_ms = (t2 - t0) * 1000 
    # 	budget_ms = self.dt * 1000
    # 	print(f"sim={sim_ms:.1f}ms  viz={viz_ms:.1f}ms  total={total_ms:.1f}ms  budget={budget_ms:.1f}ms")  
    
    # 	remaining = self.dt - (t2 - t0) 
    # 	if remaining > 0:   
    #     	time.sleep(remaining)

    def _update_visualization(self):
        pass

    def startRecording(self):
        pass

    def stopAndPublishRecording(self):
        pass

    def reset(self):
        self.simulator.Initialize()
        self.count = 0


class Go2Sim(RobotSim):
    # Joint names in pinocchio external order (indices 7-18 of get_positions())
    _JOINT_NAMES = [
        'bl_abad', 'bl_shoulder', 'bl_knee',
        'br_abad', 'br_shoulder', 'br_knee',
        'fl_abad', 'fl_shoulder', 'fl_knee',
        'fr_abad', 'fr_shoulder', 'fr_knee',
    ]

    def _update_visualization(self):
        if self.robo_robot is None:
            return

        q = self.robot.GetPositions(self.plant_context)
        
        base_pos = q[4:7]
        base_quat = q[0:4].flatten()
        
        self.robo_robot.pos = base_pos.flatten()
        qw, qx, qy, qz = base_quat
        self.robo_robot.rot = pinocchio.Quaternion(x=qx, y=qy, z=qz, w=qw).toRotationMatrix()

        # for i, name in enumerate(self._JOINT_NAMES):
        #     self.robo_robot[name] = q[7 + i]

        # Access the internal representation because iterating and setting each joint individually with 
        # `self.robo_robot[name] = angle` will trigger the forward kinematics for each joint assignment. 
        # This way, it is only triggered once
        self.robo_robot._q = q[7:]

    def set_torques(self, tau):
        tau_internal = np.zeros(12)
        tau_internal[0:6] = tau[6:12]
        tau_internal[6:12] = tau[0:6]
        self.robot.get_actuation_input_port().FixValue(self.plant_context, tau_internal)

    def get_positions(self):
        ''' Retrieve position from drake and reorder to [x, y, z, quaternion_base pos, joint_positions] '''
        q = self.robot.GetPositions(self.plant_context)
        q_external = np.zeros(19)
        q_external[0] = q[4]
        q_external[1] = q[5]
        q_external[2] = q[6]
        q_external[3] = q[1]
        q_external[4] = q[2]
        q_external[5] = q[3]
        q_external[6] = q[0]
        q_external[7:13] = q[13:19]
        q_external[13:19] = q[7:13]
        return q_external

    def get_velocities(self):
        qd = self.robot.GetVelocities(self.plant_context)
        qd_external = np.zeros(18)
        qd_external[0:3] = qd[3:6]
        qd_external[3:6] = qd[0:3]
        qd_external[6:12] = qd[12:18]
        qd_external[12:18] = qd[6:12]
        return qd_external

    def position_hold(self, q_r, duration=5.0, k_p=1000, k_d=100):
        t = 0.0
        while t < duration:
            q = self.get_positions()[7:19]
            q_d = self.get_velocities()[6:18]
            tau = k_p * (q_r - q) + k_d * (-q_d)
            self.set_torques(tau)
            self.step()
            t += self.dt
