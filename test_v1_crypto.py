"""Unit tests for aiposematic v1.0 crypto primitives and round-trip."""

import unittest
import numpy as np
from aiposematic import (
    _derive_sbox,
    _derive_diffusion_init,
    _key_rotate_right,
    _key_rotate_left,
    _apply_ops_v1,
    _apply_diffusion,
    _remove_diffusion,
    V1_OPS,
    V1_INV_OPS,
    scramble,
    recover,
    SCRAMBLE_MODE,
)
import os
import cv2
import tempfile


class TestDeriveSbox(unittest.TestCase):
    """Tests for _derive_sbox()."""

    def test_deterministic(self):
        """Same cipher_key produces same S-box."""
        fwd1, inv1 = _derive_sbox("key123")
        fwd2, inv2 = _derive_sbox("key123")
        np.testing.assert_array_equal(fwd1, fwd2)
        np.testing.assert_array_equal(inv1, inv2)

    def test_bijective(self):
        """S-box is a valid permutation (all 256 values present)."""
        fwd, inv = _derive_sbox("bijective_test")
        self.assertEqual(len(set(fwd.tolist())), 256)
        self.assertEqual(len(set(inv.tolist())), 256)

    def test_inverse(self):
        """inv_sbox[sbox[x]] == x for all x."""
        fwd, inv = _derive_sbox("inverse_test")
        for x in range(256):
            self.assertEqual(inv[fwd[x]], x)
            self.assertEqual(fwd[inv[x]], x)

    def test_different_keys(self):
        """Different cipher_keys produce different S-boxes."""
        fwd1, _ = _derive_sbox("key_a")
        fwd2, _ = _derive_sbox("key_b")
        self.assertFalse(np.array_equal(fwd1, fwd2))


class TestDiffusionInit(unittest.TestCase):
    """Tests for _derive_diffusion_init() (primitive still available)."""

    def test_deterministic(self):
        init1 = _derive_diffusion_init("test_key")
        init2 = _derive_diffusion_init("test_key")
        self.assertEqual(init1, init2)

    def test_range(self):
        init = _derive_diffusion_init("range_test")
        self.assertGreaterEqual(init, 0)
        self.assertLessEqual(init, 255)


class TestKeyRotation(unittest.TestCase):
    """Tests for _key_rotate_right/left."""

    def test_roundtrip(self):
        """rotate_left(rotate_right(a, k), k) == a for all values."""
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, size=(100, 3), dtype=np.int32)
        k = rng.integers(0, 256, size=(100, 3), dtype=np.int32)

        rotated = _key_rotate_right(a, k)
        recovered = _key_rotate_left(rotated, k)
        np.testing.assert_array_equal(recovered, a)

    def test_roundtrip_reverse(self):
        """rotate_right(rotate_left(a, k), k) == a."""
        rng = np.random.default_rng(99)
        a = rng.integers(0, 256, size=(100, 3), dtype=np.int32)
        k = rng.integers(0, 256, size=(100, 3), dtype=np.int32)

        rotated = _key_rotate_left(a, k)
        recovered = _key_rotate_right(rotated, k)
        np.testing.assert_array_equal(recovered, a)

    def test_variable_shift(self):
        """Different key values produce different rotation amounts."""
        a = np.array([[128, 128, 128]], dtype=np.int32)
        k1 = np.array([[1, 1, 1]], dtype=np.int32)
        k2 = np.array([[5, 5, 5]], dtype=np.int32)

        r1 = _key_rotate_right(a, k1)
        r2 = _key_rotate_right(a, k2)
        self.assertFalse(np.array_equal(r1, r2))


class TestOpRoundtrips(unittest.TestCase):
    """Test round-trip for each of the 9 operations."""

    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.a = self.rng.integers(0, 256, size=(200, 3), dtype=np.int32)
        self.b = self.rng.integers(0, 256, size=(200, 3), dtype=np.int32)
        self.sbox, self.inv_sbox = _derive_sbox("test_ops")

    def _test_op_roundtrip(self, op_char):
        inv_char = V1_INV_OPS[op_char]
        fwd = V1_OPS[op_char](self.a.copy(), self.b.copy(), self.sbox, self.inv_sbox)
        fwd = np.asarray(fwd).astype(np.int32)
        rec = V1_OPS[inv_char](fwd, self.b.copy(), self.sbox, self.inv_sbox)
        rec = np.asarray(rec) & 0xFF
        np.testing.assert_array_equal(rec, self.a, err_msg=f"Roundtrip failed for op '{op_char}'")

    def test_op_add(self):
        self._test_op_roundtrip('+')

    def test_op_sub(self):
        self._test_op_roundtrip('-')

    def test_op_xor(self):
        self._test_op_roundtrip('^')

    def test_op_rotate_right(self):
        self._test_op_roundtrip('>')

    def test_op_rotate_left(self):
        self._test_op_roundtrip('<')

    def test_op_permute(self):
        self._test_op_roundtrip('p')

    def test_op_inv_permute(self):
        self._test_op_roundtrip('P')

    def test_op_channel_add(self):
        self._test_op_roundtrip('a')

    def test_op_channel_sub(self):
        self._test_op_roundtrip('A')


class TestAllOpsKeyDependent(unittest.TestCase):
    """Verify each op produces different output for different keys."""

    def test_key_dependent(self):
        rng = np.random.default_rng(77)
        a = rng.integers(0, 256, size=(50, 3), dtype=np.int32)
        k1 = rng.integers(0, 256, size=(50, 3), dtype=np.int32)
        k2 = (k1 + 37) % 256

        sbox1, inv_sbox1 = _derive_sbox("key1")
        sbox2, inv_sbox2 = _derive_sbox("key2")

        for op_char in V1_OPS:
            r1 = np.asarray(V1_OPS[op_char](a.copy(), k1.copy(), sbox1, inv_sbox1))
            r2 = np.asarray(V1_OPS[op_char](a.copy(), k2.copy(), sbox2, inv_sbox2))
            self.assertFalse(
                np.array_equal(r1, r2),
                f"Op '{op_char}' produced same output for different keys"
            )


class TestPartitioning(unittest.TestCase):
    """Tests for _apply_ops_v1 partitioning model."""

    def test_roundtrip(self):
        """Partitioned op_string: inverse recovers original."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        key_pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        sbox, inv_sbox = _derive_sbox("part_test")

        for op_string in ["-^+", "+", "^>p", "-^+p>a<PA", "p", ">", "a"]:
            scrambled = _apply_ops_v1(pixels, key_pixels, op_string, sbox, inv_sbox, inverse=False)
            recovered = _apply_ops_v1(scrambled, key_pixels, op_string, sbox, inv_sbox, inverse=True)
            np.testing.assert_array_equal(
                recovered, pixels,
                err_msg=f"Roundtrip failed for op_string='{op_string}'"
            )

    def test_partitioning_assigns_one_op_per_pixel(self):
        """Verify pixel_i gets op_string[i % L], not all ops."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(6, 3)).astype(np.uint8)
        key_pixels = rng.integers(0, 256, size=(6, 3)).astype(np.uint8)
        sbox, inv_sbox = _derive_sbox("partition_test")

        # With op_string="+^", pixel 0,2,4 get '+' and pixel 1,3,5 get '^'
        result = _apply_ops_v1(pixels, key_pixels, "+^", sbox, inv_sbox)

        # Manually compute expected: even pixels get '+', odd pixels get '^'
        expected = np.empty_like(pixels, dtype=np.int32)
        for i in range(6):
            a = pixels[i:i+1].astype(np.int32)
            b = key_pixels[i:i+1].astype(np.int32)
            if i % 2 == 0:
                expected[i] = np.asarray(V1_OPS['+'](a, b, sbox, inv_sbox)).flatten()
            else:
                expected[i] = np.asarray(V1_OPS['^'](a, b, sbox, inv_sbox)).flatten()

        np.testing.assert_array_equal(result, expected.astype(np.uint8))


class TestDiffusion(unittest.TestCase):
    """Tests for block diffusion primitives (available but not used in pipeline)."""

    def test_roundtrip(self):
        """remove(apply(pixels, init), init) == pixels."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        init_byte = 42

        diffused = _apply_diffusion(pixels, init_byte)
        recovered = _remove_diffusion(diffused, init_byte)
        np.testing.assert_array_equal(recovered, pixels)

    def test_avalanche(self):
        """1-pixel change propagates to subsequent blocks."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(200, 3)).astype(np.uint8)
        init_byte = 100

        diffused1 = _apply_diffusion(pixels.copy(), init_byte)

        pixels_modified = pixels.copy()
        pixels_modified[0] = ((pixels_modified[0].astype(np.int32) + 1) % 256).astype(np.uint8)
        diffused2 = _apply_diffusion(pixels_modified, init_byte)

        self.assertFalse(np.array_equal(diffused1[:64], diffused2[:64]))
        if len(pixels) > 64:
            self.assertFalse(np.array_equal(diffused1[64:128], diffused2[64:128]))


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""

    @classmethod
    def setUpClass(cls):
        """Create a small test image."""
        cls.test_img_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(cls.test_img_path, img)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.unlink(cls.test_img_path)

    def test_scramble_recover_exact(self):
        """Full scramble->recover round-trip is pixel-perfect."""
        result = scramble(self.test_img_path, op_string="-^+")
        recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            cipher_key=result['cipher_key'],
            op_string="-^+"
        )

        original = cv2.imread(self.test_img_path)
        recovered_img = cv2.imread(recovered)
        np.testing.assert_array_equal(original, recovered_img)

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])
        os.unlink(recovered)

    def test_scramble_recover_all_ops(self):
        """Round-trip with all 9 ops partitioned."""
        op_string = "-^+p>a<PA"
        result = scramble(self.test_img_path, op_string=op_string)
        recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            cipher_key=result['cipher_key'],
            op_string=op_string
        )

        original = cv2.imread(self.test_img_path)
        recovered_img = cv2.imread(recovered)
        np.testing.assert_array_equal(original, recovered_img)

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])
        os.unlink(recovered)

    def test_wrong_cipher_key_with_sbox_ops(self):
        """Wrong cipher_key produces different output for S-box-dependent ops."""
        op_string = "p"  # S-box op depends on cipher_key
        result = scramble(self.test_img_path, op_string=op_string)

        wrong_recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            cipher_key="wrong_key_12345678",
            op_string=op_string
        )

        original = cv2.imread(self.test_img_path)
        wrong_img = cv2.imread(wrong_recovered)
        self.assertFalse(np.array_equal(original, wrong_img))

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])
        os.unlink(wrong_recovered)

    def test_wrong_opstring_garbage(self):
        """Wrong op_string produces different output."""
        result = scramble(self.test_img_path, op_string="-^+")

        wrong_recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            cipher_key=result['cipher_key'],
            op_string="+^-"
        )

        original = cv2.imread(self.test_img_path)
        wrong_img = cv2.imread(wrong_recovered)
        self.assertFalse(np.array_equal(original, wrong_img))

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])
        os.unlink(wrong_recovered)

    def test_sbox_differs_per_cipher_key(self):
        """Two scrambles with same key image but different cipher_keys produce different results for sbox ops."""
        op_string = "p>p"  # ops that depend on cipher_key (sbox + rotation)
        result1 = scramble(self.test_img_path, cipher_key="aaaa" * 8, op_string=op_string)
        result2 = scramble(self.test_img_path, cipher_key="bbbb" * 8, op_string=op_string)

        img1 = cv2.imread(result1['scrambled_path'])
        img2 = cv2.imread(result2['scrambled_path'])

        # Different cipher_keys -> different sboxes -> different scrambled output
        # (key images also differ since they're randomly generated, but the sbox
        # difference alone would cause divergence)
        self.assertFalse(np.array_equal(img1, img2))

        os.unlink(result1['scrambled_path'])
        os.unlink(result1['key_path'])
        os.unlink(result2['scrambled_path'])
        os.unlink(result2['key_path'])


if __name__ == '__main__':
    unittest.main()
