import os,sys
sys.path.append("/opt/openrobots/lib/python3.10/site-packages")
import pinocchio as pin
import numpy as np

class RobotModel:
    def __init__(self, urdf_file, floating_base=True):
        # Load URDF model into pinocchio
        if(floating_base):
            self.model = pin.buildModelFromUrdf(urdf_file, pin.JointModelFreeFlyer())
        else:
            self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()

    # Compute space Jacobian (6 x 18). 
    def spaceJacobian(self, q, frame_id_as_string):
        q_pin = self.convertPositionVector(q)
        pin.forwardKinematics(self.model, self.data, q_pin)
        # Note: The convention LOCAL_WORLD_ALIGNED means that both linear and angular part are expressed in fixed 
        # frame (world) but with the rotation point being the origin of the body frame
        return pin.computeFrameJacobian(self.model, self.data, q_pin, self.model.getFrameId(frame_id_as_string), pin.LOCAL_WORLD_ALIGNED)

    # 18 x 1 Vector of gravity and coriolis-centrifugal forces in joint space
    def biasForces(self, q):
        q_pin = self.convertPositionVector(q)
        # Ignore velocity dependent terms here!
        pin.forwardKinematics(self.model, self.data, q_pin)
        return pin.computeGeneralizedGravity(self.model, self.data,q_pin)

    # 18 x 18 mass-inertia matrix in joint space
    def massInertiaMatrix(self, q):
        q_pin = self.convertPositionVector(q)
        pin.forwardKinematics(self.model, self.data, q_pin)
        return pin.crba(self.model, self.data,q_pin)

    def pose(self, q, frame_id_as_string):
        q_pin = self.convertPositionVector(q)
        pin.forwardKinematics(self.model, self.data,q_pin)
        return pin.updateFramePlacement(self.model, self.data, self.model.getFrameId(frame_id_as_string))

    def twist(self, q, qd, frame_id_as_string):
        q_pin = self.convertPositionVector(q)
        qd_pin = self.convertVelocityVector(qd)
        pin.forwardKinematics(self.model, self.data, q_pin, qd_pin)
        pin.updateFramePlacement(self.model, self.data, self.model.getFrameId(frame_id_as_string))
        return pin.getFrameVelocity(self.model, self.data, self.model.getFrameId(frame_id_as_string), pin.LOCAL_WORLD_ALIGNED)

    def convertPositionVector(self, x):
        x_pin = pin.randomConfiguration(self.model)
        for i in range(len(x_pin)):
            x_pin[i] = x[i]
        return x_pin

    def convertVelocityVector(self, x):
        x_pin = pin.randomConfiguration(self.model)[0:18]
        for i in range(len(x_pin)):
            x_pin[i] = x[i]
        return x_pin        