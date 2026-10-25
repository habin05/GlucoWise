import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import auth_db


class AuthDbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "u.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_login_ok_and_wrong(self):
        auth_db.add_user("Alice", "secret1", "A", self.db)
        self.assertEqual(auth_db.verify_user("alice", "secret1", self.db), "A")
        self.assertEqual(auth_db.verify_user(" ALICE ", "secret1", self.db), "A")
        self.assertIsNone(auth_db.verify_user("alice", "wrong", self.db))
        self.assertIsNone(auth_db.verify_user("nobody", "secret1", self.db))

    def test_validation(self):
        with self.assertRaises(ValueError):
            auth_db.add_user("x", "secret1", "G", self.db)
        with self.assertRaises(ValueError):
            auth_db.add_user("x", "123", "A", self.db)
        with self.assertRaises(ValueError):
            auth_db.add_user("  ", "secret1", "A", self.db)

    def test_password_not_stored_plain(self):
        auth_db.add_user("bob", "secret1", "B", self.db)
        raw = self.db.read_bytes()
        self.assertNotIn(b"secret1", raw)

    def test_overwrite_and_list(self):
        auth_db.add_user("bob", "secret1", "B", self.db)
        auth_db.add_user("bob", "newpass1", "C", self.db)
        self.assertIsNone(auth_db.verify_user("bob", "secret1", self.db))
        self.assertEqual(auth_db.verify_user("bob", "newpass1", self.db), "C")
        self.assertEqual(auth_db.list_users(self.db), [("bob", "C")])


if __name__ == "__main__":
    unittest.main()
