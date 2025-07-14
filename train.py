"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import numpy as np
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from ksim.types import PhysicsData, PhysicsModel
from mujoco_animator.format import MjAnim


def hinge_angle_to_quat(theta_rad: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    """theta_rad : (..., J)   angles
    axis      : (J, 3)     unit axes (parent frame)
    returns   : (..., J, 4)
    """
    theta_rad = jnp.atleast_2d(theta_rad)  # ensure leading “time” dim
    half = 0.5 * theta_rad  # (..., J)
    w = jnp.cos(half)[..., None]  # (..., J, 1)
    xyz = jnp.sin(half)[..., None] * axis  # broadcast (..., J, 3)
    return jnp.concatenate([w, xyz], axis=-1)  # (..., J, 4)


def hinge_speed_to_omega(theta_dot: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    """theta_dot : (..., J)   angular speeds
    axis      : (J, 3)
    returns   : (..., J, 3)
    """
    theta_dot = jnp.atleast_2d(theta_dot)
    return theta_dot[..., None] * axis  # broadcast (..., J, 3)


def _get_joint_axes_in_order(mj_model: mujoco.MjModel) -> jnp.ndarray:
    """Returns array shape (num_hinge, 3) whose rows are each hinge's unit axis
    in the **parent** frame, ordered exactly like qpos[7:] / qvel[6:].
    """
    hinge_axes = []
    for j_id in range(mj_model.njnt):
        if mj_model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_axes.append(mj_model.jnt_axis[j_id])
    return jnp.asarray(hinge_axes)  # (num_hinge, 3)


# These are in the order of the neural network outputs.
JOINT_BIASES: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", 0.0),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", 0.0),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", 0.0),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", 0.0),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", 0.0),
    ("dof_right_hip_roll_03", 0.0),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", 0.0),
    ("dof_right_ankle_02", 0.0),
    ("dof_left_hip_pitch_04", 0.0),
    ("dof_left_hip_roll_03", 0.0),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", 0.0),
    ("dof_left_ankle_02", 0.0),
]

# These are in the order of the neural network outputs.
JOINT_MOTION_WEIGHTS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 1.0),
    ("dof_right_shoulder_roll_03", 100.0),
    ("dof_right_shoulder_yaw_02", 1.0),
    ("dof_right_elbow_02", 1.0),
    ("dof_right_wrist_00", 1.0),
    ("dof_left_shoulder_pitch_03", 1.0),
    ("dof_left_shoulder_roll_03", 1.0),
    ("dof_left_shoulder_yaw_02", 1.0),
    ("dof_left_elbow_02", 1.0),
    ("dof_left_wrist_00", 1.0),
    ("dof_right_hip_pitch_04", 1.0),
    ("dof_right_hip_roll_03", 1.0),
    ("dof_right_hip_yaw_03", 1.0),
    ("dof_right_knee_04", 1.0),
    ("dof_right_ankle_02", 1.0),
    ("dof_left_hip_pitch_04", 1.0),
    ("dof_left_hip_roll_03", 1.0),
    ("dof_left_hip_yaw_03", 1.0),
    ("dof_left_knee_04", 1.0),
    ("dof_left_ankle_02", 1.0),
]

JOINT_WEIGHT_ARRAY = np.asarray([w for _, w in JOINT_MOTION_WEIGHTS], dtype=np.float32)

# Hand body names for computing hand positions
LEFT_HAND_BODY_NAME = "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop"
RIGHT_HAND_BODY_NAME = "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop"


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU acceleration and gyroscope observations.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v in JOINT_BIASES}
        joint_targets = [zeros[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class CurrentQuatObservation(ksim.Observation):
    hinge_axes: xax.HashableArray

    @classmethod
    def create(cls, mj_model):
        axes = _get_joint_axes_in_order(mj_model)
        return cls(hinge_axes=xax.HashableArray(axes))

    def observe(self, state, curriculum_level, rng):
        qpos_j = state.physics_state.data.qpos[7:]
        quat = hinge_angle_to_quat(qpos_j, self.hinge_axes.array)  # (J,4)
        return quat.reshape(-1)  # flat (4 × J,)


@attrs.define(frozen=True, kw_only=True)
class CurrentJointVelocityObservation(ksim.Observation):
    hinge_axes: xax.HashableArray

    @classmethod
    def create(cls, mj_model):
        axes = _get_joint_axes_in_order(mj_model)
        return cls(hinge_axes=xax.HashableArray(axes))

    def observe(self, state, curriculum_level, rng):
        qvel_j = state.physics_state.data.qvel[6:]
        omega = hinge_speed_to_omega(qvel_j, self.hinge_axes.array)  # (J,3)
        return omega.reshape(-1)


@attrs.define(frozen=True)
class ReferenceMotionPhaseObservation(ksim.Observation):
    cycle_period: float  # seconds

    def observe(self, state, *_):
        t = jnp.atleast_1d(state.physics_state.data.time)  # (scalar or B,)
        return (t % self.cycle_period) / self.cycle_period


@attrs.define(frozen=True, kw_only=True)
class TimeDependentReferenceMotionObservation(ksim.StatefulObservation):
    """Reference motion observation."""

    qpos_arr: xax.HashableArray
    dt: float

    @classmethod
    def create(
        cls,
        *,
        physics_model: ksim.PhysicsModel,
        qpos_arr: xax.HashableArray,
        dt: float = 0.02,
    ) -> Self:
        return cls(qpos_arr=qpos_arr, dt=dt)

    def observe_stateful(
        self,
        state: ksim.ObservationInput,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        t = jnp.atleast_1d(state.physics_state.data.time)

        index = (t / self.dt).astype(jnp.int32)
        # make the motion circular
        index = jnp.mod(index + state.obs_carry, self.qpos_arr.array.shape[1])

        qpos = self.qpos_arr.array[:, index, 7:]

        return jnp.squeeze(qpos, axis=0).reshape(-1, 1), state.obs_carry

    def initial_carry(self, physics_state: ksim.PhysicsState, rng: PRNGKeyArray) -> Array:
        """Returns carry array [offset_idx]."""
        offset_idx = jax.random.randint(rng, shape=(1,), minval=0, maxval=self.qpos_arr.array.shape[1], dtype=jnp.int32)
        return offset_idx


@attrs.define(frozen=True, kw_only=True)
class TimeDependentRefQuat(ksim.StatefulObservation):
    quat_arr: xax.HashableArray  # (1, T, J, 4)
    dt: float = 0.02

    def observe_stateful(self, state, curriculum_level, rng):
        t = jnp.atleast_1d(state.physics_state.data.time)
        idx = jnp.mod((t / self.dt).astype(jnp.int32) + state.obs_carry, self.quat_arr.array.shape[1])
        quat = self.quat_arr.array[:, idx]  # (1, 1, J, 4)
        return quat.reshape(-1), state.obs_carry  # flat 4*J

    def initial_carry(self, physics_state, rng):
        return jax.random.randint(rng, (1,), 0, self.quat_arr.array.shape[1], dtype=jnp.int32)


@attrs.define(frozen=True, kw_only=True)
class TimeDependentRefVel(ksim.StatefulObservation):
    omega_arr: xax.HashableArray  # (1, T, J, 3)
    dt: float = 0.02

    def observe_stateful(self, state, curriculum_level, rng):
        t = jnp.atleast_1d(state.physics_state.data.time)
        idx = jnp.mod((t / self.dt).astype(jnp.int32) + state.obs_carry, self.omega_arr.array.shape[1])
        omega = self.omega_arr.array[:, idx]  # (1, 1, J, 3)
        return omega.reshape(-1), state.obs_carry  # flat 3*J

    def initial_carry(self, physics_state, rng):
        return jax.random.randint(rng, (1,), 0, self.omega_arr.array.shape[1], dtype=jnp.int32)


@attrs.define(frozen=True, kw_only=True)
class TimeDependentRefEEPos(ksim.StatefulObservation):
    ee_arr: xax.HashableArray  # (1,T,6)  –  [lx,ly,lz, rx,ry,rz]
    dt: float = 0.02

    def observe_stateful(self, state, _, __):
        t = jnp.atleast_1d(state.physics_state.data.time)
        idx = jnp.mod((t / self.dt).astype(jnp.int32) + state.obs_carry, self.ee_arr.array.shape[1])
        ee = self.ee_arr.array[:, idx]  # (1,1,6)
        return ee.reshape(-1), state.obs_carry

    def initial_carry(self, _, rng):
        return jax.random.randint(rng, (1,), 0, self.ee_arr.array.shape[1], jnp.int32)


@attrs.define(frozen=True, kw_only=True)
class TimeDependentRefCOM(ksim.StatefulObservation):
    com_arr: xax.HashableArray  # (1,T,3)
    dt: float = 0.02

    def observe_stateful(self, state, _, __):
        t = jnp.atleast_1d(state.physics_state.data.time)
        idx = jnp.mod((t / self.dt).astype(jnp.int32) + state.obs_carry, self.com_arr.array.shape[1])
        com = self.com_arr.array[:, idx]  # (1,1,3)
        return com.reshape(-1), state.obs_carry

    def initial_carry(self, _, rng):
        return jax.random.randint(rng, (1,), 0, self.com_arr.array.shape[1], jnp.int32)


def safe_set_data_field(data: PhysicsData, name: str, value: Array) -> PhysicsData:
    """Like ksim.update_data_field, but works for both mjx.Data and MjData."""
    if hasattr(data, "replace"):  # mjx.Data → immutable pytree
        return data.replace(**{name: value})
    else:  # mujoco.MjData → mutable struct
        setattr(data, name, np.asarray(value, dtype=data.qpos.dtype))
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomMotionFrameReset(ksim.Reset):
    """Initialises qpos/qvel from a random frame of the reference motion."""

    # Stored as static Python tuples – no HashableArray ↔ no PyTree hashing!
    positions: tuple[tuple[float, ...], ...]  # (T, nq)
    velocities: tuple[tuple[float, ...], ...]  # (T, nv)
    dt: float
    reset_pos: bool = False  # keep free-joint if False

    @classmethod
    def create(
        cls,
        *,
        motion: xax.FrozenDict,  # contains "qpos" (1,T,nq)
        physics_model: PhysicsModel,
        dt: float,
        reset_pos: bool = False,
    ) -> "RandomMotionFrameReset":
        qpos = np.asarray(motion["qpos"][0])  # (T, nq)
        nv = int(physics_model.nv)
        qvel = np.zeros((qpos.shape[0], nv), qpos.dtype)  # zero ω/ẋ

        # convert to nested tuples ⇒ static / hashable
        pos_tuple = tuple(map(tuple, qpos.tolist()))
        vel_tuple = tuple(map(tuple, qvel.tolist()))
        return cls(positions=pos_tuple, velocities=vel_tuple, dt=float(dt), reset_pos=reset_pos)

    def __call__(
        self,
        data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PhysicsData:
        # convert back to JAX arrays (cheap – happens once per reset)
        pos_arr = jnp.asarray(self.positions)
        vel_arr = jnp.asarray(self.velocities)

        frame = jax.random.randint(rng, (), 0, pos_arr.shape[0])

        # slice reference frame
        qpos_ref = pos_arr[frame]
        qvel_ref = vel_arr[frame]

        # keep or overwrite the free joint
        if self.reset_pos:
            new_qpos = qpos_ref
        else:
            new_qpos = jnp.concatenate([data.qpos[:7], qpos_ref[7:]])

        data = safe_set_data_field(data, "qpos", new_qpos)
        data = safe_set_data_field(data, "qvel", qvel_ref)

        # always set time in a JIT-friendly way
        new_time = frame.astype(jnp.float32) * self.dt
        data = safe_set_data_field(data, "time", new_time)

        return data


@attrs.define(frozen=True)
class ReferenceMotionReward(ksim.Reward):
    """Reward for tracking the reference motion."""

    reference_motion_obs_name: str = attrs.field(default="time_dependent_reference_motion_observation")
    joint_scales: tuple[float, ...] = attrs.field(
        default=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    )

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reference_motion = trajectory.obs[self.reference_motion_obs_name]
        reference_motion = jnp.squeeze(reference_motion, axis=-1)

        joint_pos = trajectory.qpos[..., 7:]

        # Compute the difference between the reference motion and the current joint positions
        norm = xax.get_norm(reference_motion - joint_pos, norm="l2")
        return (ksim.norm_to_reward(norm) * jnp.array(self.joint_scales)).sum(axis=-1)


# ---------- 1. joint-orientation (quaternion) ------------------------------
@attrs.define(frozen=True)
class MimicJointOrientationReward(ksim.Reward):
    joint_weights: xax.HashableArray
    quat_ref_name: str = "time_dependent_ref_quat"
    quat_cur_name: str = "current_quat_observation"

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        weights = jnp.asarray(self.joint_weights.array)

        # reshape into (B, J, 4) instead of the old flat vector
        ref = traj.obs[self.quat_ref_name].reshape(traj.done.shape[0], 20, 4)
        cur = traj.obs[self.quat_cur_name].reshape(traj.done.shape[0], 20, 4)

        # per-joint error (B, J)
        per_joint_err = jnp.linalg.norm(ref - cur, axis=-1)  # e_j

        # turn each joint’s error into a score first
        per_joint_score = jnp.exp(-2.0 * per_joint_err)  # α ≈ 2 originally

        # now weight the score, not the exponent
        weighted = per_joint_score * weights

        # aggregate – e.g. mean or sum
        r = weighted.mean(-1)  # (B,)
        return jnp.broadcast_to(r, traj.done.shape)


# ---------- 2. joint-angular-velocity --------------------------------------


@attrs.define(frozen=True)
class MimicJointVelocityReward(ksim.Reward):
    joint_weights: xax.HashableArray
    vel_ref_name: str = "time_dependent_ref_vel"
    vel_cur_name: str = "current_joint_velocity_observation"

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        weights = jnp.asarray(self.joint_weights.array)

        ref = traj.obs[self.vel_ref_name].reshape(traj.done.shape[0], 20, 3)
        cur = traj.obs[self.vel_cur_name].reshape(traj.done.shape[0], 20, 3)

        # per-joint error (B, J)
        per_joint_err = jnp.linalg.norm(ref - cur, axis=-1)  # e_j

        # turn each joint’s error into a score first
        per_joint_score = jnp.exp(-0.1 * per_joint_err)  # α ≈ 2 originally

        # now weight the score, not the exponent
        weighted = per_joint_score * weights

        # aggregate – e.g. mean or sum
        r = weighted.mean(-1)  # (B,)
        return jnp.broadcast_to(r, traj.done.shape)


# ---------- 3. end-effector pose -------------------------------------------
@attrs.define(frozen=True)
class MimicEEPoseReward(ksim.Reward):
    left_hand_body_id: int
    right_hand_body_id: int
    ref_ee_name: str = "time_dependent_ref_eepos"
    alpha: float = 10.0  # tracking sharpness, not a global weight

    @property
    def ee_ids(self):
        return (self.left_hand_body_id, self.right_hand_body_id)

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        if self.ref_ee_name not in traj.obs:
            return jnp.zeros(traj.done.shape)

        root_pos = traj.xpos[..., 0, :]  # (B,3)
        ee_cur = traj.xpos[..., self.ee_ids, :] - root_pos[:, None, :]
        ee_ref = traj.obs[self.ref_ee_name].reshape(*ee_cur.shape)

        error = jnp.linalg.norm(ee_ref - ee_cur, axis=-1).mean(-1)  # (B,)
        r = jnp.exp(-self.alpha * error)  # no extra scale
        return jnp.broadcast_to(r, traj.done.shape)


# ---------- 4. centre-of-mass (root) tracking ------------------------------
@attrs.define(frozen=True)
class MimicCOMReward(ksim.Reward):
    ref_com_name: str = "time_dependent_ref_com"  # reference traj key
    cur_com_name: str = "base_position_observation"  # current robot root pos

    def get_reward(self, traj: ksim.Trajectory) -> Array:
        # make sure both obs exist
        if self.ref_com_name not in traj.obs or self.cur_com_name not in traj.obs:
            return jnp.zeros(traj.done.shape)

        ref = traj.obs[self.ref_com_name].reshape(*traj.obs[self.cur_com_name].shape)
        cur = traj.obs[self.cur_com_name]  # (B,3)

        err = jnp.linalg.norm(ref - cur, axis=-1)  # (B,)
        r = jnp.exp(-10.0 * err)
        return jnp.broadcast_to(r, traj.done.shape)


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v in JOINT_BIASES])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def __init__(self, config: HumanoidWalkingTaskConfig) -> None:
        super().__init__(config)

        # Get hand body IDs - needed for computing hand positions in motion data
        self.mj_model = self.get_mujoco_model()
        self.hinge_axes = self._get_joint_axes_in_order(self.mj_model)
        self.hand_left_id = ksim.get_body_data_idx_from_name(self.mj_model, LEFT_HAND_BODY_NAME)
        self.hand_right_id = ksim.get_body_data_idx_from_name(self.mj_model, RIGHT_HAND_BODY_NAME)

        self.real_motion = self.get_real_motions(self.mj_model)  # FrozenDict with qpos/quat/omega/hand_pos
        self.cycle_period = self.real_motion["qpos"].shape[1] * self.config.ctrl_dt
        self.motion_weights = xax.HashableArray(JOINT_WEIGHT_ARRAY)

    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def _get_joint_axes_in_order(self, mj_model: mujoco.MjModel) -> jnp.ndarray:
        """Returns array shape (num_hinge, 3) whose rows are each hinge's unit axis
        in the **parent** frame, ordered exactly like qpos[7:] / qvel[6:].
        """
        hinge_axes = []
        for j_id in range(mj_model.njnt):
            if mj_model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_HINGE:
                hinge_axes.append(mj_model.jnt_axis[j_id])
        return jnp.asarray(hinge_axes)  # (num_hinge, 3)

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_qpos_from_file(self, filepath: Path) -> Array:
        if filepath.suffix == ".npz":
            npz = np.load(filepath, allow_pickle=True)
            qpos = jnp.array(npz["qpos"])[400:]  # skip first 200 frames
            if float(npz["frequency"]) != 1 / self.config.ctrl_dt:
                raise ValueError(f"Motion frequency {npz['frequency']} does not match ctrl_dt {self.config.ctrl_dt}")
            return qpos

        # Load the JSON animation file
        anim = MjAnim.load_json(filepath)

        # Convert to numpy array with proper time stepping
        # This gives us shape (T, num_dofs) where T is number of timesteps
        qpos = anim.to_numpy(dt=self.config.ctrl_dt, interp="linear", loop=True)

        # Verify the motion frequency matches our control timestep
        # Calculate total duration from the animation frames
        total_duration = sum(frame.length for frame in anim.frames)
        expected_frames = int(total_duration / self.config.ctrl_dt)
        logging.info(
            "Animation duration: %.3fs, Expected frames: %d, Actual frames: %d"
            % (total_duration, expected_frames, qpos.shape[0])
        )
        return qpos

    def get_real_motions(self, mj_model: mujoco.MjModel) -> PyTree:
        """Loads a trajectory from a .npz file and converts it to the (batch, T, 20) tensor expected by AMP.

        Expected keys inside the .npz:
          • 'qpos'            –  (T, Nq)
          • optional 'frequency' –  sampling Hz (used for verification)
        """
        # traj_path = Path(__file__).parent / "gaits" / "cmu_walking_91.npz"
        # traj_path = Path(__file__).parent / "gaits" / "dance_salsa.npz"
        traj_path = Path(__file__).parent / "motions" / "basic_arm.json"

        qpos = self.get_qpos_from_file(traj_path)

        # npz = np.load(traj_path, allow_pickle=True)
        # qpos = jnp.array(npz["qpos"])[400:] # skip first 200 frames

        # pos_limits = ksim.get_position_limits(mj_model)

        # jnp.clip(arr, pos_limits)

        # if float(npz["frequency"]) != 1 / self.config.ctrl_dt:
        #     raise ValueError(f"Motion frequency {npz['frequency']} does not match ctrl_dt {self.config.ctrl_dt}")

        # Create data for forward kinematics
        mj_data = mujoco.MjData(mj_model)

        # Compute hand positions for each timestep
        t = qpos.shape[0]
        hand_pos = np.zeros((t, 6))  # (T, [left_x, left_y, left_z, right_x, right_y, right_z])
        hand_abs = np.zeros((t, 6))  # absolute hand positions
        com_world = np.zeros((t, 3))  # center-of-mass positions

        for t in range(t):
            # Set the qpos for this timestep
            mj_data.qpos = qpos[t]
            # Forward kinematics to compute body positions
            mujoco.mj_forward(mj_model, mj_data)

            # Get root position and orientation
            root_pos = mj_data.xpos[0]  # Assuming root is at index 0

            # Get hand positions in world frame
            left_hand_world = mj_data.xpos[self.hand_left_id]
            right_hand_world = mj_data.xpos[self.hand_right_id]

            # Convert to root-relative positions
            left_hand_rel = left_hand_world - root_pos
            right_hand_rel = right_hand_world - root_pos

            # Store local coordinates
            hand_pos[t, 0:3] = left_hand_rel
            hand_pos[t, 3:6] = right_hand_rel

            # --- NEW: absolute hand positions (world frame) --------------------------
            left_hand_w = mj_data.xpos[self.hand_left_id].copy()
            right_hand_w = mj_data.xpos[self.hand_right_id].copy()
            hand_abs[t, 0:3] = left_hand_w  # (T,6)
            hand_abs[t, 3:6] = right_hand_w

            # --- NEW: centre-of-mass position (world) --------------------------------
            com_world[t] = mj_data.subtree_com[0]  # (T,3)

        # Convert to jax array and add batch dimension
        hand_pos = jnp.array(hand_pos)[None]
        hand_abs = jnp.array(hand_abs)[None]  # (1,T,6)
        com_world = jnp.array(com_world)[None]  # (1,T,3)

        joint_limits = ksim.get_position_limits(mj_model)
        joint_names = ksim.get_joint_names_in_order(mj_model)

        joint_mins = []
        joint_maxs = []
        for name in joint_names[1:]:  # skip freejoint
            if name not in joint_limits:
                raise KeyError(f"Joint '{name}' missing from joint limits dictionary")
            j_min, j_max = joint_limits[name]
            joint_mins.append(j_min)
            joint_maxs.append(j_max)

        joint_mins_arr = jnp.asarray(joint_mins)
        joint_maxs_arr = jnp.asarray(joint_maxs)

        # Separate freejoint (7) and articulated joints.
        qpos_root = qpos[:, :7]
        qpos_joints = qpos[:, 7:]

        # Bring each angle into range by shifting with multiples of 2π.
        two_pi = 2.0 * math.pi
        center_arr = (joint_mins_arr + joint_maxs_arr) / 2.0  # (J,)

        # Vectorised 2π-shifting about the joint-range centre.
        qpos_orig = qpos_joints  # keep a copy for statistics
        qpos_shifted = qpos_orig - jnp.round((qpos_orig - center_arr) / two_pi) * two_pi

        # Final clipping (handles ranges narrower than 2π or numerical drift).
        qpos_joints = jnp.clip(qpos_shifted, joint_mins_arr[None, :], joint_maxs_arr[None, :])

        # Re-assemble the full qpos.
        qpos = jnp.concatenate([qpos_root, qpos_joints], axis=-1)

        # -----------------------------------------------------------
        # ### NEW: build reference quaternions and angular velocities
        # -----------------------------------------------------------
        # 1.  finite-difference to get hinge angular speed θ̇  (rad/s)
        dt = self.config.ctrl_dt
        qvel_joints = jnp.zeros_like(qpos_joints).at[1:].set((qpos_joints[1:] - qpos_joints[:-1]) / dt)  # shape (T, J)

        # 2.  broadcast helpers across (T,J)
        #     self.hinge_axes  : (J, 3)  – collect once in __init__
        quat_T_J_4 = hinge_angle_to_quat(
            qpos_joints,  # (T, J, 1)
            self.hinge_axes,  # (J, 3)
        )  # -> (T, J, 4)

        omega_T_J_3 = hinge_speed_to_omega(
            qvel_joints,  # (T, J, 1)
            self.hinge_axes,  # (J, 3)
        )  # -> (T, J, 3)

        # 3.  add batch dim
        quat = quat_T_J_4[None]  # (1, T, J, 4)
        omega = omega_T_J_3[None]  # (1, T, J, 3)
        # -----------------------------------------------------------

        # add batch dim to qpos & hand_pos just like you already do
        qpos = qpos[None]  # (1, T, Nq)
        # hand_pos  = jnp.array(hand_pos)[None]

        # --------------------------------------------
        # Return everything the rest of your pipeline
        # might need (old + new keys)
        # --------------------------------------------
        return xax.FrozenDict(
            {
                "qpos": qpos,  # (1, T, Nq)
                "hand_pos": hand_pos,  # (1, T, 6) - root-relative
                "hand_abs": hand_abs,  # (1, T, 6) - absolute world frame
                "quat": quat,  # (1, T, J, 4)   ### NEW
                "omega": omega,  # (1, T, J, 3)   ### NEW
                "com": com_world,  # (1, T, 3) - center of mass world frame
            }
        )

    def motion_to_qpos(self, motion: PyTree) -> Array:
        """Converts a motion to `qpos` array.

        This function is just used for replaying the motion on the robot model
        for visualization purposes.

        Args:
            motion: The full motion, including the batch dimension.

        Returns:
            The `qpos` array, with shape (B, T, N).
        """
        return motion["qpos"]

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            # ksim.PushEvent(
            #     x_force=1.0,
            #     y_force=1.0,
            #     z_force=0.3,
            #     force_range=(0.5, 1.0),
            #     x_angular_force=0.0,
            #     y_angular_force=0.0,
            #     z_angular_force=0.0,
            #     interval_range=(0.5, 4.0),
            # ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            # ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            RandomMotionFrameReset.create(
                motion=self.real_motion,  # already loaded
                physics_model=physics_model,
                dt=self.config.ctrl_dt,
                reset_pos=False,  # keep root pose
            ),
            # ksim.RandomJointPositionReset.create(physics_model, {k: v for k in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=1.0,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
            CurrentQuatObservation.create(mj_model=self.mj_model),
            CurrentJointVelocityObservation.create(mj_model=self.mj_model),
            ReferenceMotionPhaseObservation(cycle_period=self.cycle_period),
            TimeDependentReferenceMotionObservation.create(
                physics_model=physics_model,
                qpos_arr=xax.HashableArray(self.real_motion["qpos"]),
            ),
            TimeDependentRefQuat(
                quat_arr=xax.HashableArray(self.real_motion["quat"]),
                dt=self.config.ctrl_dt,
            ),
            TimeDependentRefVel(
                omega_arr=xax.HashableArray(self.real_motion["omega"]),
                dt=self.config.ctrl_dt,
            ),
            TimeDependentRefEEPos(
                ee_arr=xax.HashableArray(self.real_motion["hand_pos"]),
                dt=self.config.ctrl_dt,
            ),
            TimeDependentRefCOM(
                com_arr=xax.HashableArray(self.real_motion["com"]),
                dt=self.config.ctrl_dt,
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.StayAliveReward(scale=10.0),
            # ReferenceMotionReward(scale=1.0),
            # ----- Deep-Mimic imitation terms (now split) ----------------------
            MimicJointOrientationReward(scale=0.65, joint_weights=self.motion_weights),
            MimicJointVelocityReward(scale=0.10, joint_weights=self.motion_weights),
            MimicEEPoseReward(
                left_hand_body_id=self.hand_left_id,
                right_hand_body_id=self.hand_right_id,
                scale=0.15,  # EE       w = 0.15
            ),
            MimicCOMReward(scale=0.10),  # COM      w = 0.10
            # # Standard rewards.
            # ksim.NaiveForwardReward(clip_max=1.25, in_robot_frame=False, scale=3.0),
            # ksim.NaiveForwardOrientationReward(scale=1.0),
            # ksim.StayAliveReward(scale=1.0),
            # ksim.UprightReward(scale=0.5),
            # # Avoid movement penalties.
            # ksim.AngularVelocityPenalty(index=("x", "y"), scale=-0.1),
            # ksim.LinearVelocityPenalty(index=("z"), scale=-0.1),
            # # Normalization penalties.
            # ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01),
            # ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.JointJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.ActionAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # # Bespoke rewards.
            # BentArmPenalty.create_penalty(physics_model, scale=-0.1),
            # StraightLegPenalty.create_penalty(physics_model, scale=-0.1),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.DistanceFromOriginCurriculum(
            min_level_steps=5,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_actor_inputs=(50 if self.config.use_acc_gyro else 44) + 20,
            num_actor_outputs=len(JOINT_BIASES),
            num_critic_inputs=(445 if self.config.use_acc_gyro else 401) + 20,
            min_std=0.001,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        phase_1 = observations["reference_motion_phase_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        ref_motion = observations["time_dependent_reference_motion_observation"].squeeze(-1)  # (20,)

        obs = [
            phase_1,  # 1
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
            ]

        obs += [
            ref_motion,  # (NUM_JOINTS)
        ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        phase_1 = observations["reference_motion_phase_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        ref_motion = observations["time_dependent_reference_motion_observation"].squeeze(-1)  # (20,)

        obs_n = jnp.concatenate(
            [
                phase_1,  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                ref_motion,  # (NUM_JOINTS)
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _model_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._model_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            global_grad_clip=2.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            render_full_every_n_seconds=300,
            valid_every_n_seconds=300,
            render_azimuth=145.0,
        ),
    )
