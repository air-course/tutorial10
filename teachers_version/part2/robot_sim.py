from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization
from pydrake.geometry import StartMeshcat,MeshcatParams,Meshcat,HalfSpace,MeshcatVisualizer
from pydrake.common import FindResourceOrThrow
from IPython.display import display, HTML
from pydrake.systems.framework import LeafSystem, Context, BasicVector
from pydrake.math import RotationMatrix, RigidTransform

import matplotlib.pyplot as plt

import os
import subprocess
import sys
import numpy as np
import time

def install_deepnote_nginx():
    """Uses Ubuntu to install the NginX web server and configures it to serve
    as a reverse proxy for MeshCat on Deepnote. The server will proxy
    https://DEEPNOTE_PROJECT_ID:8080/PORT/ to http://127.0.0.1:PORT/ so
    that multiple notebooks can all be served via Deepnote's only open port.
    """
    print("Installing NginX server for MeshCat on Deepnote...")
    install_nginx = FindResourceOrThrow(
        "drake/setup/deepnote/install_nginx")
    proc = subprocess.run(
        [install_nginx], encoding="utf-8", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    if proc.returncode == 0:
        return
    print(proc.stdout, file=sys.stderr, end="")
    proc.check_returncode()

class RobotSim:
    def __init__(self, meshcat, urdf_file, q_0, dt):
        self.dt = dt
        self.builder = DiagramBuilder()
        self.robot, self.scene_graph = AddMultibodyPlantSceneGraph(builder=self.builder, time_step=dt)
        Parser(self.robot, self.scene_graph).AddModels(urdf_file)
        self.q_0 = q_0
        self.AddGround(self.robot)
        self.robot.Finalize()
        self.robot.SetDefaultPositions(self.q_0)

        host = os.environ["DEEPNOTE_PROJECT_ID"]
        port = 7000
        url = "https://" + host + ".deepnoteproject.com/" + str(port) + "/"
        self.meshcat = meshcat
        self.visualizer = MeshcatVisualizer.AddToBuilder(self.builder,self.scene_graph, self.meshcat)
        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()

        self.simulator = Simulator(self.diagram, self.diagram_context)
        self.simulator.Initialize()
        self.simulator.set_publish_every_time_step(False)
        self.simulator.set_target_realtime_rate(1.0)
        self.plant_context = self.robot.GetMyContextFromRoot(self.simulator.get_mutable_context())

        self.count = 0

    def AddGround(self,plant):
        """
        Add a flat ground with friction
        """

        # Constants
        transparent_color = np.array([0.5,0.5,0.5,0])
        nontransparent_color = np.array([0.5,0.5,0.5,0.1])

        p_GroundOrigin = [0, 0.0, 0.0]
        R_GroundOrigin = RotationMatrix.MakeXRotation(0.0)
        X_GroundOrigin = RigidTransform(R_GroundOrigin,p_GroundOrigin)

        # Set Up Ground on Plant

        surface_friction = CoulombFriction(
                static_friction = 0.7,
                dynamic_friction = 0.5)
        plant.RegisterCollisionGeometry(
                plant.world_body(),
                X_GroundOrigin,
                HalfSpace(),
                "ground_collision",
                surface_friction)
        plant.RegisterVisualGeometry(
                plant.world_body(),
                X_GroundOrigin,
                HalfSpace(),
                "ground_visual",
                transparent_color)  # transparent
        
    def getJointNames(self):
        return self.robot.GetActuatorNames()

    def step(self):
        self.simulator.AdvanceTo(self.dt*self.count)
        self.count += 1
        time.sleep(self.dt)
    
    def startRecording(self):
        self.visualizer.StartRecording()
    
    def stopAndPublishRecording(self):
        self.visualizer.StopRecording()
        self.visualizer.PublishRecording()

    def reset(self):
        self.visualizer.DeleteRecording()
        self.simulator.Initialize()
        
class Go2Sim(RobotSim):
    def set_torques(self,tau):
        tau_internal = np.zeros(12)
        tau_internal[0:6] = tau[6:12]
        tau_internal[6:12] = tau[0:6]
        self.robot.get_actuation_input_port().FixValue(self.plant_context, tau_internal)

    def get_positions(self):
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
    
    def position_hold(self, q_r, duration = 5.0, k_p = 1000, k_d = 100):
        t = 0.0
        while t < duration:
            q   = self.get_positions()[7:19]
            q_d = self.get_velocities()[6:18]
            tau = k_p * (q_r - q) + k_d * (-q_d)
            self.set_torques(tau)
            self.step()
            t += self.dt
