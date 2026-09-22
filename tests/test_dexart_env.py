import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from dexcg.envs.dexart import DexArtAdapter
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS


class Actor:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class Pose:
    def to_transformation_matrix(self) -> np.ndarray:
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, 2.0, 3.0]
        return matrix


def observation_with_state(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "instance_1-rgb": np.zeros((2, 2, 3), dtype=np.float32),
        "instance_1-depth": np.zeros((2, 2), dtype=np.float32),
        "instance_1-point_cloud": np.zeros((4, 6), dtype=np.float32),
        "instance_1-seg_gt": np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        ),
        "instance_1-object_center": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "imagination_robot": np.zeros((96, 7), dtype=np.float32),
        "state": state,
    }


def model_observation(raw):
    identity = SimpleNamespace(to_transformation_matrix=lambda: np.eye(4))
    adapter = SimpleNamespace(
        environment=SimpleNamespace(
            robot=SimpleNamespace(get_pose=lambda: identity, get_qpos=lambda: raw["state"][:22]),
            palm_link=SimpleNamespace(get_pose=lambda: identity),
        )
    )
    adapter.robot_proprioception = lambda: DexArtAdapter.robot_proprioception(adapter)
    return DexArtAdapter.observation(adapter, raw)


def test_observation_inserts_missing_joint_before_time_progress() -> None:
    state = np.arange(32, dtype=np.float32)

    sample = model_observation(observation_with_state(state))

    np.testing.assert_array_equal(sample["state"], state)
    np.testing.assert_array_equal(sample["object_point_mask"], [True, True, False, False])
    np.testing.assert_allclose(sample["object_center"], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(sample["agent_pos"][:31], state[:31])
    assert sample["agent_pos"][31] == 0.0
    assert sample["agent_pos"][32] == state[31]


def test_observation_preserves_native_33d_agent_position() -> None:
    state = np.arange(33, dtype=np.float32)

    sample = model_observation(observation_with_state(state))

    expected = state.copy()
    expected[31] = 0
    np.testing.assert_array_equal(sample["state"], expected)
    np.testing.assert_array_equal(sample["agent_pos"], expected)


def test_robot_proprioception_uses_live_pose_and_is_invariant_to_world_frame() -> None:
    from scipy.spatial.transform import Rotation

    def transform(rotation, position):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
        result[:3, 3] = position
        return result

    world_from_base = transform([0.3, -0.6, 0.9], [1.0, -2.0, 3.0])
    base_from_palm = transform([-0.2, 0.4, 0.5], [0.3, 0.1, 0.6])
    world_from_palm = world_from_base @ base_from_palm
    qpos = np.linspace(-0.5, 0.5, 22, dtype=np.float32)
    links = [Actor(link.dexart_link) for link in ALLEGRO_CONTACT_LINKS]
    environment = SimpleNamespace(
        robot=SimpleNamespace(
            get_links=lambda: links,
            get_qpos=lambda: qpos,
            get_pose=lambda: SimpleNamespace(to_transformation_matrix=lambda: world_from_base),
        ),
        palm_link=SimpleNamespace(
            get_pose=lambda: SimpleNamespace(to_transformation_matrix=lambda: world_from_palm)
        ),
        instance_links=[],
    )
    adapter = DexArtAdapter(environment)
    first = adapter.robot_proprioception()
    np.testing.assert_allclose(first["palm_pose_robot_base"], base_from_palm, atol=1e-6)
    np.testing.assert_array_equal(first["robot_qpos"], qpos)
    palm_point = np.array([0.03, -0.02, 0.01, 1.0])
    np.testing.assert_allclose(
        world_from_base @ first["palm_pose_robot_base"] @ palm_point,
        world_from_palm @ palm_point,
        atol=1e-6,
    )
    change_world = transform([0.4, 0.5, -0.8], [-2.0, 1.0, 0.4])
    world_from_base = change_world @ world_from_base
    world_from_palm = change_world @ world_from_palm
    qpos[:] += 0.1
    second = adapter.robot_proprioception()
    np.testing.assert_allclose(second["palm_pose_robot_base"], base_from_palm, atol=1e-6)
    np.testing.assert_allclose(second["robot_qpos"] - first["robot_qpos"], 0.1, atol=1e-6)


def test_contact_graph_filters_impulse_and_averages_in_robot_frame() -> None:
    links = [Actor(link.dexart_link) for link in ALLEGRO_CONTACT_LINKS]
    object_link = Actor("object")
    weak_point = SimpleNamespace(
        position=np.array([9.0, 9.0, 9.0]), impulse=np.array([0.001, 0, 0])
    )
    points = [
        SimpleNamespace(position=np.array([2.0, 3.0, 4.0]), impulse=np.array([0.02, 0, 0])),
        SimpleNamespace(position=np.array([4.0, 5.0, 6.0]), impulse=np.array([0.02, 0, 0])),
    ]
    contacts = [
        SimpleNamespace(actor0=links[1], actor1=object_link, points=[weak_point]),
        SimpleNamespace(actor0=links[0], actor1=object_link, points=points),
    ]
    environment = SimpleNamespace(
        robot=SimpleNamespace(get_links=lambda: links, get_pose=lambda: Pose()),
        instance_links=[object_link],
        scene=SimpleNamespace(get_contacts=lambda: contacts),
    )

    graph = DexArtAdapter(environment, impulse_threshold=1.0e-2).contact_graph()

    assert graph.mask.sum() == 1
    assert graph.mask[0]
    np.testing.assert_allclose(graph.points[0], [2.0, 2.0, 2.0])


def test_create_wraps_explicit_object_id_in_single_element_list(monkeypatch) -> None:
    links = [Actor(link.dexart_link) for link in ALLEGRO_CONTACT_LINKS]
    environment = SimpleNamespace(
        robot=SimpleNamespace(get_links=lambda: links),
        instance_links=[],
    )
    received = {}

    def create_env(**kwargs):
        received.update(kwargs)
        return environment

    create_env_module = ModuleType("dexart.env.create_env")
    create_env_module.create_env = create_env
    task_setting_module = ModuleType("dexart.env.task_setting")
    task_setting_module.TRAIN_CONFIG = {"faucet": {"seen": [148, 693]}}
    task_setting_module.RANDOM_CONFIG = {"faucet": {"rand_pos": 0.1, "rand_degree": 90}}
    monkeypatch.setitem(sys.modules, "dexart.env.create_env", create_env_module)
    monkeypatch.setitem(sys.modules, "dexart.env.task_setting", task_setting_module)

    adapter = DexArtAdapter.create("faucet", "seen", 0.01, object_id=693)

    assert adapter.environment is environment
    assert received["index"] == [693]


def test_create_rejects_object_outside_requested_split(monkeypatch) -> None:
    create_env_module = ModuleType("dexart.env.create_env")
    create_env_module.create_env = lambda **kwargs: None
    task_setting_module = ModuleType("dexart.env.task_setting")
    task_setting_module.TRAIN_CONFIG = {"faucet": {"seen": [148, 693]}}
    task_setting_module.RANDOM_CONFIG = {"faucet": {"rand_pos": 0.1, "rand_degree": 90}}
    monkeypatch.setitem(sys.modules, "dexart.env.create_env", create_env_module)
    monkeypatch.setitem(sys.modules, "dexart.env.task_setting", task_setting_module)

    with np.testing.assert_raises_regex(ValueError, "not in the faucet seen split"):
        DexArtAdapter.create("faucet", "seen", 0.01, object_id=1556)
