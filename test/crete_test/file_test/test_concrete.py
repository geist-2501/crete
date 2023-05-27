import os
import unittest

from crete.file.concrete import ConcreteFile
from .. import test_dir


class TestConcrete(unittest.TestCase):
    def test_reading_and_writing(self):
        file = ConcreteFile(
            id="abc",
            agent_data="hello!".encode(),
            training_artifacts={},
            config={}
        )

        file_path = os.path.join(test_dir, "test_concfile.conc")
        file.write(file_path)
        read_back_file: ConcreteFile = ConcreteFile.read(file_path)

        agent_data = read_back_file.agent_data.decode()
        self.assertEqual(agent_data, "hello!")


if __name__ == '__main__':
    unittest.main()
