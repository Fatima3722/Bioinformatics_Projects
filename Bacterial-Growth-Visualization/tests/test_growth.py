import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'plot_growth.py'
spec = importlib.util.spec_from_file_location('growth', SCRIPT)
growth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(growth)


class GrowthChecks(unittest.TestCase):
    def test_ordering_and_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.csv'
            path.write_text('Time_Hours,Strain,OD600\n2,A,0.2\n0,A,0.05\n')
            frame, keys = growth.load_growth(path)
            self.assertEqual(frame.Time_Hours.tolist(), [0, 2])
            output = growth.plot_growth(path, Path(temp)/'plot.png')
            self.assertGreater(output.stat().st_size, 1000)

    def test_bad_inputs_and_replicate_identifiers(self):
        invalid = ['Time_Hours,Strain,OD600\n0,A,inf\n',
                   'Time_Hours,Strain,OD600\n0,,0.1\n',
                   'Time_Hours,Strain,OD600\n0,A,0.1\n0,A,0.2\n',
                   'Time_Hours,OD600\n0,0.1\n']
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.csv'
            for text in invalid:
                path.write_text(text)
                with self.assertRaises(ValueError):
                    growth.load_growth(path)
            path.write_text('Time_Hours,Strain,OD600,Replicate\n0,A,0.1,1\n0,A,0.2,2\n')
            self.assertEqual(growth.load_growth(path)[1], ['Strain', 'Replicate'])

    def test_example_and_missing_input_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'example.png'
            good = subprocess.run([sys.executable, str(SCRIPT), '--example', '--output', str(output)],
                                  capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertTrue(output.is_file())
            bad = subprocess.run([sys.executable, str(SCRIPT), '--input', str(Path(temp)/'missing.csv')],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
