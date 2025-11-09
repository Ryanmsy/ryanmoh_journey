# test_calc.py

import unittest
from unittest.mock import patch
import calc
from employee import Employee

#what does TestCase do
class Testcalc(unittest.TestCase):


    def test_add(self):
    # normal case
    self.assertEqual(calc.add(10, 5), 15)

    # edge-ish case
    self.assertEqual(calc.add(10, -2), 8)

def test_divide(self):
    self.assertEqual(calc.divide(10, 2), 5)

    # two ways to test exceptions
    self.assertRaises(ValueError, calc.divide, 10, 0)

    with self.assertRaises(ValueError):
        calc.divide(10, 0)


class TestEmployee(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # runs once before all tests in this class
        print("setUpClass")

    @classmethod
    def tearDownClass(cls):
        # runs once after all tests in this class
        print("tearDownClass")

    def setUp(self):
        # runs before every single test_... method
        self.emp_1 = Employee('Ryan', 'Moh', 5000)
        self.emp_2 = Employee('Jane', 'Doe', 6000)

    def tearDown(self):
        # runs after every test_... method
        pass

    def test_email(self):
        # initial email
        self.assertEqual(self.emp_1.email, 'RyanMoh@email.com')

        # change first name → email should update because it's @property
        self.emp_1.first = "Jon"
        self.assertEqual(self.emp_1.email, 'JonMoh@email.com')

    def test_apply_raise(self):
        self.emp_1.apply_raise()
        # 5000 * 1.04 = 5200
        self.assertEqual(self.emp_1.pay, 5200)

    def test_monthly_schedule(self):
        # success case
        with patch('employee.requests.get') as mocked_get:
            mocked_get.return_value.ok = True
            mocked_get.return_value.text = 'Success'

            schedule = self.emp_1.monthly_schedule('May')

            # check the correct URL was called
            mocked_get.assert_called_with('http://company.com/Moh/May')
            self.assertEqual(schedule, 'Success')

        # failure case
        with patch('employee.requests.get') as mocked_get:
            mocked_get.return_value.ok = False

            schedule = self.emp_2.monthly_schedule('June')

            mocked_get.assert_called_with('http://company.com/Doe/June')
            self.assertEqual(schedule, 'Bad response')


if __name__ == '__main__':
    unittest.main()