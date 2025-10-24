from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bandaid import BandAidPlacer


@pytest.fixture
def placer(tmp_path):
    """Provide a fresh BandAidPlacer with its band-aid image isolated per test."""
    bandaid_path = tmp_path / "test_bandaid.png"
    # Ensure each test works with its own generated asset
    instance = BandAidPlacer(str(bandaid_path))
    return instance


def make_landmarks(mp_pose, shoulder, elbow, wrist):
    """Construct a minimal landmark list covering shoulder, elbow, and wrist."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0) for _ in range(33)]
    indices = {
        mp_pose.PoseLandmark.RIGHT_SHOULDER: shoulder,
        mp_pose.PoseLandmark.RIGHT_ELBOW: elbow,
        mp_pose.PoseLandmark.RIGHT_WRIST: wrist,
        mp_pose.PoseLandmark.LEFT_SHOULDER: shoulder,
        mp_pose.PoseLandmark.LEFT_ELBOW: elbow,
        mp_pose.PoseLandmark.LEFT_WRIST: wrist,
    }
    for idx, (x, y) in indices.items():
        landmarks[int(idx)] = SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)
    return SimpleNamespace(landmark=landmarks)


def test_calculate_bandaid_transform_right_arm(placer):
    # Synthetic landmark positions (normalized)
    right_landmarks = make_landmarks(
        placer.mp_pose,
        shoulder=(0.5, 0.4),
        elbow=(0.6, 0.5),
        wrist=(0.7, 0.6),
    )

    transform = placer.calculate_bandaid_transform(right_landmarks, (1000, 800, 3), arm_side="right")

    assert transform["position"] == (512, 540)
    assert transform["angle"] == pytest.approx(51.34, abs=0.1)
    assert transform["scale"] == pytest.approx(0.32, abs=0.01)


def test_calculate_bandaid_transform_left_arm_clamps_scale(placer):
    left_landmarks = make_landmarks(
        placer.mp_pose,
        shoulder=(0.5, 0.4),
        elbow=(0.4, 0.5),
        wrist=(0.3, 0.55),
    )

    transform = placer.calculate_bandaid_transform(left_landmarks, (1000, 800, 3), arm_side="left")

    assert transform["position"] == (288, 520)
    assert transform["angle"] == pytest.approx(147.99, abs=0.1)
    assert transform["scale"] == 0.3  # Clamped minimum


def test_overlay_bandaid_crops_without_shifting(placer):
    # Replace the real band-aid with a small synthetic RGBA patch
    patch = np.zeros((4, 4, 4), dtype=np.uint8)
    patch[:, :, 0] = 200  # Red channel
    patch[:, :, 3] = 128  # 50% alpha
    placer.bandaid_img = patch

    base_image = np.zeros((10, 10, 3), dtype=np.uint8)
    transform = {"position": (1, 5), "angle": 0.0, "scale": 1.0}

    result = placer.overlay_bandaid(base_image, transform)

    # Overlap should tint columns 0-2 starting at row 3, without being shifted inside
    tinted_region = result[3:7, 0:3, 0]
    assert np.allclose(tinted_region, 100, atol=1)
    assert np.all(result[:, 3, :] == 0)


def test_process_image_pipeline_success(monkeypatch, tmp_path, placer):
    image_path = tmp_path / "dummy.jpg"
    dummy_image = np.zeros((50, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), dummy_image)

    fake_landmarks = make_landmarks(
        placer.mp_pose,
        shoulder=(0.5, 0.4),
        elbow=(0.6, 0.5),
        wrist=(0.7, 0.6),
    )

    monkeypatch.setattr(placer, "detect_arm", lambda img: fake_landmarks)

    recorded = {}

    def fake_overlay(image, transform):
        recorded["transform"] = transform
        return np.full_like(image, 255)

    monkeypatch.setattr(placer, "overlay_bandaid", fake_overlay)

    expected_transform = placer.calculate_bandaid_transform(fake_landmarks, dummy_image.shape, arm_side="right")

    original, result, detected = placer.process_image(str(image_path), arm_side="right", show_landmarks=False)

    assert detected is True
    assert np.array_equal(original, dummy_image)
    assert np.array_equal(result, np.full_like(dummy_image, 255))
    assert recorded["transform"]["position"] == expected_transform["position"]
    assert recorded["transform"]["scale"] == pytest.approx(expected_transform["scale"], abs=1e-6)
    assert recorded["transform"]["angle"] == pytest.approx(expected_transform["angle"], abs=1e-6)


def test_process_image_no_landmarks(monkeypatch, tmp_path, placer):
    image_path = tmp_path / "dummy.jpg"
    dummy_image = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), dummy_image)

    monkeypatch.setattr(placer, "detect_arm", lambda img: None)

    original, result, detected = placer.process_image(str(image_path), arm_side="right", show_landmarks=False)

    assert detected is False
    assert np.array_equal(original, result)