"""Geographic source corrections must conserve geometry and bound recipe work."""

from contextlib import redirect_stderr
from io import StringIO
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from source_landmarks import LandmarkLayer, clip_wall
from raster import Raster


def city(structures=None, facades=None):
    return {
        "origin": {"lat": 0, "lon": 0},
        "photographic": {
            "crs": 3857,
            "landmark_geometry": {
                "structures": structures or [],
                "facades": facades or [],
            },
        },
    }


def evidence(identifier):
    return {
        "id": identifier,
        "source_refs": ["https://example.org/survey"],
        "interpretation": "Synthetic surveyed example; decorative detail is illustrative.",
    }


def obelisk():
    return dict(
        evidence("test-obelisk"),
        kind="tapered_obelisk",
        anchor={"lat": 0, "lon": 0},
        replace_owners=[7],
        yaw_degrees=0,
        height_m=30,
        shaft_height_m=24,
        base_width_m=8,
        upper_width_m=4,
        bands=[[0, 12, [180, 170, 150]], [12, 24, [237, 234, 220]]],
        windows={"z_band": [20, 22], "centres_m": [-0.8, 0.8], "half_width_m": 0.3},
        flags={"count": 2, "radius_m": 12, "height_m": 4},
    )


def colonnade():
    return dict(
        evidence("test-colonnade"),
        kind="arc_colonnade",
        anchor={"lat": 0, "lon": 0},
        yaw_degrees=0,
        offset_m=[0, 0],
        radius_m=5,
        columns=4,
        base_z_m=10,
        top_z_m=18,
        decks=[[17.5, 0.5, [240, 240, 230]], [13, 0.2, [235, 235, 225]]],
        balustrade={"count": 8, "bottom_z_m": 13.2, "top_z_m": 14},
    )


def facade():
    return dict(
        evidence("test-wall"),
        owners=[8],
        wall_rgb=[160, 80, 60],
        anchor={"lat": 0, "lon": 0},
        support_box_m=[-10, -10, 10, 10],
        window_phase_m=0,
        window_spacing_m=4,
        window_width_m=[1, 2],
        window_z_bands_m=[[10, 12], [14, 16]],
        mullion_phase_m=1.5,
        opening={"box_m": [-3, -3, 3, 3], "keep_below_z_m": 2},
    )


class Camera:
    width, height = 128, 160

    def project(self, points):
        x, y, z = np.asarray(points).T
        return np.column_stack(
            [64 + 2 * (x - y), 150 - 2 * (z + 0.4 * (x + y)), x + y + 0.2 * z]
        )


class RecordingRaster(Raster):
    def __init__(self):
        super().__init__(Camera(), {"background_rgb": (0, 0, 0)})
        self.surfaces = []

    def triangle(self, xyz, color, *args, **kwargs):
        self.surfaces.append((np.asarray(xyz), color, kwargs.get("shader")))
        return super().triangle(xyz, color, *args, **kwargs)


class SourceGeometry(unittest.TestCase):
    def test_obelisk_is_tapered_survey_geometry_with_only_observation_openings(self):
        raster = RecordingRaster()
        layer = LandmarkLayer(city([obelisk()]))
        layer.validate_owners(np.array([7, 9], dtype=np.uint64))
        report = layer.draw(raster, lambda x, y: 10)[0]
        vertices = np.asarray(report["vertices"])
        np.testing.assert_allclose(vertices[:4, 2], 10)
        np.testing.assert_allclose(vertices[4:8, 2], 34)
        np.testing.assert_allclose(vertices[-1], [0, 0, 40])
        np.testing.assert_allclose(np.ptp(vertices[:4, :2], axis=0), [8, 8])
        np.testing.assert_allclose(np.ptp(vertices[4:8, :2], axis=0), [4, 4])
        self.assertTrue(np.isfinite(raster.depth).any())
        # Sample the actual material callback at many lower-shaft positions.
        # Its flat stone must not grow the general facade's repeating office windows.
        shader = next(shader for _, _, shader in raster.surfaces if shader is not None)
        lower = np.array(
            [
                [x, -3, z]
                for x in np.linspace(-2, 2, 21)
                for z in np.linspace(11, 29, 19)
            ]
        )
        rgb = shader(lower, np.array([180, 170, 150]), None, None)
        np.testing.assert_array_equal(rgb, np.tile([180, 170, 150], (len(lower), 1)))
        samples = np.array(
            [[-0.8, -2.2, 31], [0.8, -2.2, 31], [-2, -2.2, 31], [2, -2.2, 31]]
        )
        rgb = shader(samples, np.array([237, 234, 220]), None, None)
        np.testing.assert_array_equal(rgb[:2], [[55, 60, 57], [55, 60, 57]])
        np.testing.assert_array_equal(rgb[2:], [[237, 234, 220], [237, 234, 220]])

    def test_clip_conserves_plane_winding_area_and_linear_camera_depth(self):
        spec = facade()
        for triangle, fraction in [
            (np.array([[-1.0, 0, 0], [1, 0, 4], [0, 2, 4]]), 0.25),
            (np.array([[-1.0, 0, 0], [1, 0, 0], [0, 2, 4]]), 0.75),
        ]:
            with self.subTest(triangle=triangle.tolist()):
                fragments = clip_wall(triangle, np.zeros(2), spec)
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                area = 0
                for fragment in fragments:
                    self.assertTrue((fragment[:, 2] <= 2).all())
                    cross = np.cross(
                        fragment[1] - fragment[0], fragment[2] - fragment[0]
                    )
                    self.assertGreater(np.dot(cross, normal), 0)
                    area += np.linalg.norm(cross) / 2
                    for point in fragment:
                        barycentric = np.linalg.lstsq(
                            np.vstack([triangle.T, np.ones(3)]),
                            np.r_[point, 1],
                            rcond=None,
                        )[0]
                        self.assertTrue((barycentric >= -1e-12).all())
                        np.testing.assert_allclose(
                            barycentric @ triangle, point, atol=1e-12
                        )
                        np.testing.assert_allclose(
                            Camera().project([point])[0, 2],
                            barycentric @ Camera().project(triangle)[:, 2],
                            atol=1e-12,
                        )
                self.assertAlmostEqual(area, np.linalg.norm(normal) / 2 * fraction)

    def test_clip_unchanged_and_empty_cases(self):
        triangle = np.array([[-1.0, 0, 0], [1, 0, 0], [0, 2, 1]])
        np.testing.assert_array_equal(
            clip_wall(triangle, np.zeros(2), facade())[0], triangle
        )
        self.assertEqual(clip_wall(triangle + [0, 0, 4], np.zeros(2), facade()), [])
        outside = triangle + [30, 0, 4]
        np.testing.assert_array_equal(
            clip_wall(outside, np.zeros(2), facade())[0], outside
        )

    def test_ownership_defaults_and_recipe_snapshot(self):
        recipe = city([obelisk()], [facade()])
        layer = LandmarkLayer(recipe)
        recipe["photographic"]["landmark_geometry"]["facades"][0]["wall_rgb"] = [
            0,
            0,
            0,
        ]
        self.assertEqual(layer.replaced, {7})
        layer.validate_owners(np.array([7, 8, 10], dtype=np.int64))
        with self.assertRaises(ValueError):
            layer.validate_owners(np.array([7]))
        for invalid in [np.array([7.0, 8.0]), np.array([7, -8]), np.array([[7, 8]])]:
            with (
                self.subTest(invalid_owners=invalid.tolist()),
                self.assertRaises(ValueError),
            ):
                layer.validate_owners(invalid)
        triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(layer.fragments(10, triangle)[0], triangle)
        self.assertEqual(layer.wall_colour(10, (1, 2, 3)), (1, 2, 3))
        default = lambda *args: np.tile([1, 2, 3], (len(args[0]), 1))
        self.assertIs(layer.shader(10, np.array([0, 1, 0]), 0, default), default)
        shaded = layer.shader(8, np.array([0, 1, 0]), 0, default)
        rgb = shaded(
            np.array([[0, 0, 13], [30, 0, 13]]),
            np.array([1, 2, 3]),
            None,
            np.array([1, 1]),
        )
        self.assertGreater(rgb[0, 0], 100)
        np.testing.assert_array_equal(rgb[1], [1, 2, 3])

    def test_plain_material_removes_office_windows_without_mutating_vertices(self):
        rule = dict(evidence("plain-surveyed-stone"), owners=[9], wall_rgb=[240,239,230], finish="plain")
        layer = LandmarkLayer(city(facades=[rule]))
        world = np.array([[0.,0.,12.],[1.,0.,18.],[2.,0.,24.]])
        original = world.copy()
        def office_grid(*args):
            self.fail("plain source stone must not call the generic office-window painter")
        shader = layer.shader(9,np.array([0.,1.,0.]),0,office_grid)
        result = shader(world,np.array([210,209,200]),None,None)
        np.testing.assert_array_equal(result,np.tile([210,209,200],(3,1)))
        np.testing.assert_array_equal(world,original)

    def test_empty_config_does_not_require_city_geometry_or_allocate_work(self):
        for photographic in [
            {},
            {"landmark_geometry": {}},
            {"landmark_geometry": {"structures": [], "facades": []}},
        ]:
            layer = LandmarkLayer({"photographic": photographic})
            layer.validate_owners(np.array([], dtype=np.uint64))
            self.assertEqual(
                layer.draw(None, lambda *args: self.fail("ground was sampled")), []
            )
            self.assertEqual(layer.wall_colour(5, (1, 2, 3)), (1, 2, 3))

    def test_nonfinite_terrain_fails_before_obelisk_is_drawn(self):
        raster = RecordingRaster()
        with self.assertRaises(ValueError):
            LandmarkLayer(city([obelisk()])).draw(raster, lambda *args: float("nan"))
        self.assertEqual(raster.surfaces, [])


class RecipeValidation(unittest.TestCase):
    def rejected(self, structures=None, facades=None):
        with self.assertRaises(ValueError):
            LandmarkLayer(city(structures, facades))

    def test_bad_structure_fields_numbers_and_shapes(self):
        mutations = [
            ("kind", "unknown"),
            ("kind", []),
            ("height_m", float("inf")),
            ("shaft_height_m", float("nan")),
            ("height_m", 20),
            ("base_width_m", 10**1000),
            ("upper_width_m", -1),
            ("bands", [[0, 10, [1, 2, 3]], [11, 24, [4, 5, 6]]]),
            ("bands", [[0, 24, [1, 2, 300]]]),
            ("bands", [[0, 24, [1, 2]]]),
            ("bands", [[0, 12, [1, 2, 3]]]),
            ("flags", {"count": 1000000, "radius_m": 12, "height_m": 4}),
            ("flags", {"count": True, "radius_m": 12, "height_m": 4}),
            ("flags", {"count": 1, "radius_m": -1, "height_m": 4}),
            (
                "windows",
                {"z_band": [22, 20], "centres_m": [-0.8, 0.8], "half_width_m": 0.3},
            ),
            ("windows", {"z_band": [20, 22], "centres_m": [4], "half_width_m": 0.3}),
            ("anchor", {"lat": 91, "lon": 0}),
            ("anchor", {"lat": 0, "lon": 3}),
            ("source_refs", []),
            ("unexpected", 1),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=str(value)[:100]):
                spec = obelisk()
                spec[key] = value
                self.rejected([spec])

    def test_bad_colonnade_and_combined_resource_budget(self):
        for key, value in [
            ("columns", 100000),
            ("offset_m", [1]),
            ("radius_m", 0),
            ("top_z_m", 11),
            ("decks", [[18, 1, [1, 2, 3]]]),
            ("balustrade", {"count": 257, "bottom_z_m": 12, "top_z_m": 14}),
        ]:
            with self.subTest(key=key):
                spec = colonnade()
                spec[key] = value
                self.rejected([spec])
        specs = []
        for index in range(4):
            spec = colonnade()
            spec["id"] += str(index)
            spec["columns"] = 128
            spec["balustrade"]["count"] = 256
            specs.append(spec)
        self.rejected(specs)
        self.rejected([dict(colonnade(), id=str(i)) for i in range(33)])

    def test_ambiguous_owners_and_ids_fail(self):
        self.rejected([obelisk(), obelisk()])
        self.rejected([dict(obelisk(), replace_owners=[7, 7])])
        self.rejected([obelisk()], [dict(facade(), owners=[7])])
        self.rejected(facades=[facade(), dict(facade(), id="second")])
        self.rejected([dict(obelisk(), replace_owners=[True])])

    def test_bad_facade_intervals_shapes_and_partial_config(self):
        for key, value in [
            ("wall_rgb", [1.0, 2, 3]),
            ("support_box_m", [0, 0, -1, 1]),
            ("window_width_m", [3, 5]),
            ("window_spacing_m", 0),
            ("window_z_bands_m", [[10, 14], [12, 16]]),
            ("opening", {"box_m": [-1, -1, 1, 1], "keep_below_z_m": float("nan")}),
        ]:
            with self.subTest(key=key):
                spec = facade()
                spec[key] = value
                self.rejected(facades=[spec])
        spec = facade()
        del spec["window_z_bands_m"]
        self.rejected(facades=[spec])
        spec = facade()
        spec["owners"] = []
        self.rejected(facades=[spec])
        self.rejected(facades=[dict(facade(),finish="plain")])
        self.rejected(facades=[dict(evidence("bad-finish"),owners=[8],wall_rgb=[1,2,3],finish="unknown")])

    def test_bad_config_crs_and_origin(self):
        for config in [None, [], {"structures": {}}, {"unknown": []}]:
            recipe = city()
            recipe["photographic"]["landmark_geometry"] = config
            with self.assertRaises(ValueError):
                LandmarkLayer(recipe)
        for crs in [4326, "bad-crs"]:
            recipe = city([obelisk()])
            recipe["photographic"]["crs"] = crs
            with self.assertRaises(ValueError):
                LandmarkLayer(recipe)
        recipe = city([obelisk()])
        recipe["origin"]["lon"] = float("inf")
        with self.assertRaises(ValueError):
            LandmarkLayer(recipe)

    def test_cli_rejects_invalid_geometry_before_creating_output(self):
        from photographic_patch import main

        recipe = city([dict(obelisk(), height_m=float("nan"))])
        recipe["id"] = "invalid-source"
        with tempfile.TemporaryDirectory(prefix="landmark-invalid-") as temporary:
            root = Path(temporary)
            path = root / "recipe.json"
            path.write_text(json.dumps(recipe))
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as failure:
                main(["--city", str(path), "--out", str(root / "output")])
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
