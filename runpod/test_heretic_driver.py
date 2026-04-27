import unittest

from heretic_driver import _choose_trial_option, _should_select_save_action


TRIAL_MENU = """
Which trial do you want to use?
[1] [Trial   2] Refusals:  4/100, KL divergence: 0.1923
[2] [Trial  24] Refusals: 32/100, KL divergence: 0.1669
[3] [Trial   1] Refusals: 38/100, KL divergence: 0.1238
[4] [Trial  10] Refusals: 44/100, KL divergence: 0.1075
[5] [Trial  20] Refusals: 52/100, KL divergence: 0.0546
[6] [Trial  23] Refusals: 54/100, KL divergence: 0.0158
[7] [Trial  21] Refusals: 71/100, KL divergence: 0.0057
[8] [Trial   6] Refusals: 79/100, KL divergence: 0.0032
[9] [Trial  16] Refusals: 83/100, KL divergence: 0.0006
[10] Run additional trials
[11] Exit program
Enter number:
"""


class HereticDriverParsingTests(unittest.TestCase):
    def test_choose_trial_lowest_refusals(self):
        self.assertEqual(_choose_trial_option(TRIAL_MENU, "lowest_refusals"), "1")

    def test_choose_trial_lowest_kl(self):
        self.assertEqual(_choose_trial_option(TRIAL_MENU, "lowest_kl"), "9")

    def test_choose_trial_best_balance(self):
        # With current weighted heuristic, the first row wins in this sample.
        self.assertEqual(_choose_trial_option(TRIAL_MENU, "best_balance"), "1")

    def test_choose_trial_default_when_no_rows(self):
        self.assertEqual(_choose_trial_option("No menu here", "best_balance"), "1")

    def test_detect_save_menu_direct_prompt(self):
        text = """
        What do you want to do with the decensored model?
        [1] Save the model to a local folder
        Enter number:
        """
        self.assertTrue(_should_select_save_action(text))

    def test_detect_save_menu_with_wrapped_text_and_ansi(self):
        text = (
            "\x1b[36mWhat do you want to do with the decensored\n"
            "model?\x1b[0m\n[1] Save the model to a local folder\nEnter number: "
        )
        self.assertTrue(_should_select_save_action(text))

    def test_detect_save_menu_by_options_only(self):
        text = """
        [1] Save the model to a local folder
        [2] Upload the model to Hugging Face
        Enter number:
        """
        self.assertTrue(_should_select_save_action(text))

    def test_detect_save_menu_false(self):
        self.assertFalse(_should_select_save_action("Which trial do you want to use?"))


if __name__ == "__main__":
    unittest.main()
