"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass
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
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from pathlib import Path

import numpy as np
from mujoco_animator.format import MjAnim

from jaxtyping import Array, PRNGKeyArray, PyTree


# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(90.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-90.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-50.0)),
    ("dof_right_ankle_02", math.radians(30.0)),
    ("dof_left_hip_pitch_04", math.radians(20.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(50.0)),
    ("dof_left_ankle_02", math.radians(-30.0)),
]


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
        zeros = {k: v for k, v in ZEROS}
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
    

@attrs.define(frozen=True)
class ReferenceMotionReward(ksim.Reward):
    """Reward for tracking the reference motion."""

    reference_motion_obs_name: str = attrs.field(default="time_dependent_reference_motion_observation")
    joint_scales: tuple[float, ...] = attrs.field(default=
    (1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0,
     0.5, 0.5, 0.5, 0.5, 0.5,
     0.5, 0.5, 0.5, 0.5, 0.5))

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        reference_motion = trajectory.obs[self.reference_motion_obs_name]
        reference_motion = jnp.squeeze(reference_motion, axis=-1)

        joint_pos = trajectory.qpos[..., 7:]

        # Compute the difference between the reference motion and the current joint positions
        norm = xax.get_norm(reference_motion - joint_pos, norm="l2")
        return (ksim.norm_to_reward(norm) * jnp.array(self.joint_scales)).sum(axis=-1)


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
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

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
        mj_model = self.get_mujoco_model()
        self.hand_left_id = ksim.get_body_data_idx_from_name(mj_model, LEFT_HAND_BODY_NAME)
        self.hand_right_id = ksim.get_body_data_idx_from_name(mj_model, RIGHT_HAND_BODY_NAME)

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
            qpos = jnp.array(npz["qpos"])[400:] # skip first 200 frames
            if float(npz["frequency"]) != 1 / self.config.ctrl_dt:
                raise ValueError(f"Motion frequency {npz['frequency']} does not match ctrl_dt {self.config.ctrl_dt}")
            return qpos
        
        from mujoco_animator.format import MjAnim
        
        # Load the JSON animation file
        anim = MjAnim.load_json(filepath)
        
        # Convert to numpy array with proper time stepping
        # This gives us shape (T, num_dofs) where T is number of timesteps
        qpos = anim.to_numpy(dt=self.config.ctrl_dt, interp="linear", loop=True)
        
        # Verify the motion frequency matches our control timestep
        # Calculate total duration from the animation frames
        total_duration = sum(frame.length for frame in anim.frames)
        expected_frames = int(total_duration / self.config.ctrl_dt)
        print(f"Animation duration: {total_duration}s, Expected frames: {expected_frames}, Actual frames: {qpos.shape[0]}")
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

        # Convert to jax array and add batch dimension
        hand_pos = jnp.array(hand_pos)[None]

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

        qpos = qpos[None]

        # Return both qpos and hand positions
        return xax.FrozenDict({"qpos": qpos, "hand_pos": hand_pos})

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
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
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
            TimeDependentReferenceMotionObservation.create(
                physics_model=physics_model,
                qpos_arr=xax.HashableArray(self.get_real_motions(self.get_mujoco_model())["qpos"]),
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.StayAliveReward(scale=10.0),
            ReferenceMotionReward(scale=1.0),
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
            num_actor_inputs=51 if self.config.use_acc_gyro else 45,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=446,
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
        time_1 = observations["timestep_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        ref_motion = observations["time_dependent_reference_motion_observation"]

        obs = [
            jnp.sin(time_1),
            jnp.cos(time_1),
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
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
        time_1 = observations["timestep_observation"]
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
        
        ref_motion = observations["time_dependent_reference_motion_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.sin(time_1),
                jnp.cos(time_1),
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
        ),
    )
