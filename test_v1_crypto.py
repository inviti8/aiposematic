"""Unit tests for aiposematic v1.1 crypto primitives and round-trip."""

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
    OP_INTENSITY,
    scramble,
    recover,
    SCRAMBLE_MODE,
    derive_cipher_key,
    derive_subscriber_cipher_key,
)
import os
import cv2
import tempfile

try:
    from hvym_stellar import Stellar25519KeyPair
    from stellar_sdk import Keypair as StellarKeypair
    _HAS_STELLAR = True
except ImportError:
    _HAS_STELLAR = False


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


class TestPerOpIntensity(unittest.TestCase):
    """Tests for v1.1 per-op intensity scaling."""

    @classmethod
    def setUpClass(cls):
        cls.test_img_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(cls.test_img_path, img)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.unlink(cls.test_img_path)

    def test_single_op_roundtrip(self):
        """Each single-op roundtrip is pixel-perfect at its intensity."""
        for op_char in OP_INTENSITY:
            result = scramble(self.test_img_path, op_string=op_char)
            recovered = recover(
                result['scrambled_path'],
                key_img_path=result['key_path'],
                cipher_key=result['cipher_key'],
                op_string=op_char,
            )
            original = cv2.imread(self.test_img_path)
            rec_img = cv2.imread(recovered)
            np.testing.assert_array_equal(
                original, rec_img,
                err_msg=f"Roundtrip failed for single op '{op_char}'"
            )
            os.unlink(result['scrambled_path'])
            os.unlink(result['key_path'])
            os.unlink(recovered)

    def test_intensity_gradient(self):
        """Mild ops produce less disruption than strong ops."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        key_pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        sbox, inv_sbox = _derive_sbox("intensity_test")

        # Mild op '>' (intensity 0.15) vs strong op '+' (intensity 1.0)
        mild = _apply_ops_v1(pixels, key_pixels, ">", sbox, inv_sbox)
        strong = _apply_ops_v1(pixels, key_pixels, "+", sbox, inv_sbox)

        mild_diff = np.abs(mild.astype(float) - pixels.astype(float)).mean()
        strong_diff = np.abs(strong.astype(float) - pixels.astype(float)).mean()

        self.assertLess(mild_diff, strong_diff,
                       f"Mild op '>' (diff={mild_diff:.1f}) should produce less disruption than '+' (diff={strong_diff:.1f})")

    def test_partitioned_intensity_roundtrip(self):
        """Mixed intensity op_string roundtrip is pixel-perfect."""
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        key_pixels = rng.integers(0, 256, size=(500, 3)).astype(np.uint8)
        sbox, inv_sbox = _derive_sbox("mixed_test")

        for op_string in [">a", "p<", "+>-<", "-^+p>a<PA"]:
            scrambled = _apply_ops_v1(pixels, key_pixels, op_string, sbox, inv_sbox, inverse=False)
            recovered = _apply_ops_v1(scrambled, key_pixels, op_string, sbox, inv_sbox, inverse=True)
            np.testing.assert_array_equal(
                recovered, pixels,
                err_msg=f"Intensity roundtrip failed for op_string='{op_string}'"
            )

    def test_rotation_max_shift_scaling(self):
        """Rotation ops with low intensity use fewer shift bits."""
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, size=(100, 3), dtype=np.int32)
        k = rng.integers(0, 256, size=(100, 3), dtype=np.int32)

        # max_shift=2 (intensity 0.15 → max(2, round(7*0.15))=max(2,1)=2)
        r_mild = _key_rotate_right(a, k, max_shift=2)
        # max_shift=7 (intensity 1.0 → max(2, round(7*1.0))=7)
        r_full = _key_rotate_right(a, k, max_shift=7)

        # Both should be valid 8-bit values
        self.assertTrue(np.all(r_mild >= 0))
        self.assertTrue(np.all(r_mild <= 255))
        self.assertTrue(np.all(r_full >= 0))
        self.assertTrue(np.all(r_full <= 255))

        # And they should generally differ
        self.assertFalse(np.array_equal(r_mild, r_full))


class TestStellarKeyDerivation(unittest.TestCase):
    """Tests for Stellar keypair-based cipher key derivation."""

    @unittest.skipUnless(_HAS_STELLAR, "hvym-stellar not installed")
    def test_derive_cipher_key_deterministic(self):
        """Same keypair produces same cipher_key."""
        kp = Stellar25519KeyPair(StellarKeypair.random())
        key1 = derive_cipher_key(kp)
        key2 = derive_cipher_key(kp)
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 32)

    @unittest.skipUnless(_HAS_STELLAR, "hvym-stellar not installed")
    def test_different_keypairs_different_keys(self):
        """Different keypairs produce different cipher_keys."""
        kp1 = Stellar25519KeyPair(StellarKeypair.random())
        kp2 = Stellar25519KeyPair(StellarKeypair.random())
        self.assertNotEqual(derive_cipher_key(kp1), derive_cipher_key(kp2))

    @unittest.skipUnless(_HAS_STELLAR, "hvym-stellar not installed")
    def test_ecdh_symmetric(self):
        """ECDH derivation is symmetric: artist→subscriber == subscriber→artist."""
        artist_kp = Stellar25519KeyPair(StellarKeypair.random())
        subscriber_kp = Stellar25519KeyPair(StellarKeypair.random())

        key_from_artist = derive_subscriber_cipher_key(artist_kp, subscriber_kp.public_key())
        key_from_subscriber = derive_subscriber_cipher_key(subscriber_kp, artist_kp.public_key())

        self.assertEqual(key_from_artist, key_from_subscriber)
        self.assertEqual(len(key_from_artist), 32)


class TestStellarIntegration(unittest.TestCase):
    """Integration tests for Stellar keypair-based scramble/recover."""

    @classmethod
    def setUpClass(cls):
        cls.test_img_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(cls.test_img_path, img)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_img_path):
            os.unlink(cls.test_img_path)

    @unittest.skipUnless(_HAS_STELLAR, "hvym-stellar not installed")
    def test_stellar_self_recovery(self):
        """Artist scramble + self-recover via same keypair is pixel-perfect."""
        kp = Stellar25519KeyPair(StellarKeypair.random())
        result = scramble(self.test_img_path, op_string="-^+", stellar_keypair=kp)
        recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            op_string="-^+",
            stellar_keypair=kp,
        )

        original = cv2.imread(self.test_img_path)
        rec_img = cv2.imread(recovered)
        np.testing.assert_array_equal(original, rec_img)

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])
        os.unlink(recovered)

    @unittest.skipUnless(_HAS_STELLAR, "hvym-stellar not installed")
    def test_ecdh_subscriber_roundtrip(self):
        """ECDH: artist scrambles for subscriber, subscriber recovers."""
        artist_kp = Stellar25519KeyPair(StellarKeypair.random())
        subscriber_kp = Stellar25519KeyPair(StellarKeypair.random())

        # Artist scrambles for subscriber
        result = scramble(
            self.test_img_path,
            op_string="-^+",
            stellar_keypair=artist_kp,
            subscriber_public_key=subscriber_kp.public_key(),
        )

        # Subscriber recovers using their own keypair + artist's public key
        recovered = recover(
            result['scrambled_path'],
            key_img_path=result['key_path'],
            op_string="-^+",
            stellar_keypair=subscriber_kp,
            artist_public_key=artist_kp.public_key(),
        )

        original = cv2.imread(self.test_img_path)
        rec_img = cv2.imread(recovered)
        np.testing.assert_array_equal(original, rec_img)

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])
        os.unlink(recovered)

    def test_no_key_generates_random(self):
        """Scramble without cipher_key or stellar_keypair generates a random key."""
        result = scramble(self.test_img_path, op_string="-^+")
        self.assertIsNotNone(result['cipher_key'])
        self.assertEqual(len(result['cipher_key']), 32)  # secrets.token_hex(16) = 32 chars

        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])

    def test_recover_without_key_raises(self):
        """Recover without any key source raises ValueError."""
        result = scramble(self.test_img_path, op_string="-^+")
        with self.assertRaises(ValueError):
            recover(
                result['scrambled_path'],
                key_img_path=result['key_path'],
                op_string="-^+",
            )
        os.unlink(result['scrambled_path'])
        os.unlink(result['key_path'])


if __name__ == '__main__':
    unittest.main()
