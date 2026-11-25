import unittest
from unittest.mock import patch

from audit.run_comfy_cpu import result_from_history, wait_for


class ComfyCPURunnerTests(unittest.TestCase):
    def test_terminal_failure_is_not_treated_as_pending_or_success(self):
        self.assertIsNone(result_from_history({}, 'p'))
        with self.assertRaisesRegex(RuntimeError, 'execution failed.*fixture error'):
            result_from_history({'p': {'status': {'status_str': 'error', 'completed': False,
                                                  'messages': ['fixture error']}}}, 'p')
        success = {'status': {'status_str': 'success', 'completed': True}}
        self.assertIs(result_from_history({'p': success}, 'p'), success)

    def test_pending_prompt_times_out(self):
        with patch('audit.run_comfy_cpu.time.monotonic', side_effect=[0, 0, 0, 0, 2]):
            with patch('audit.run_comfy_cpu.time.sleep'):
                with self.assertRaises(TimeoutError):
                    wait_for(lambda _: None, 1)
