import unittest

from value import Value


class TestValue(unittest.TestCase):
  def test_add(self):
    v1 = Value(2)
    v2 = Value(3)
    result = v1 + v2
    self.assertAlmostEqual(result.data, 5)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '+')

  def test_add_scalar(self):
    v1 = Value(2)
    result = v1 + 3
    self.assertAlmostEqual(result.data, 5)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '+')

  def test_sub(self):
    v1 = Value(5)
    v2 = Value(3)
    result = v1 - v2
    self.assertAlmostEqual(result.data, 2)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '-')

  def test_sub_scalar(self):
    v1 = Value(5)
    result = v1 - 3
    self.assertAlmostEqual(result.data, 2)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '-')

  def test_mul(self):
    v1 = Value(2)
    v2 = Value(3)
    result = v1 * v2
    self.assertAlmostEqual(result.data, 6)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '*')

  def test_mul_scalar(self):
    v1 = Value(2)
    result = v1 * 3
    self.assertAlmostEqual(result.data, 6)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '*')

  def test_truediv(self):
    v1 = Value(6)
    v2 = Value(3)
    result = v1 / v2
    self.assertAlmostEqual(result.data, 2)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '/')

  def test_truediv_scalar(self):
    v1 = Value(6)
    result = v1 / 3
    self.assertAlmostEqual(result.data, 2)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '/')

  def test_pow(self):
    v1 = Value(2)
    v2 = Value(3)
    result = v1**v2
    self.assertAlmostEqual(result.data, 8)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '^')

  def test_pow_scalar(self):
    v1 = Value(2)
    result = v1**3
    self.assertAlmostEqual(result.data, 8)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '^')

  def test_tanh(self):
    v1 = Value(1)
    result = v1.tanh()
    self.assertAlmostEqual(result.data, 0.7615941559557649)
    self.assertEqual(len(result.operands), 1)
    self.assertEqual(result.op, 'tanh')

  def test_rmul(self):
    v1 = Value(2)
    result = 3 * v1
    self.assertAlmostEqual(result.data, 6)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '*')

  def test_radd(self):
    v1 = Value(2)
    result = 3 + v1
    self.assertAlmostEqual(result.data, 5)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '+')

  def test_rsub(self):
    v1 = Value(2)
    result = 3 - v1
    self.assertAlmostEqual(result.data, 1)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '-')

  def test_rtruediv(self):
    v1 = Value(2)
    result = 6 / v1
    self.assertAlmostEqual(result.data, 3)
    self.assertEqual(len(result.operands), 2)
    self.assertEqual(result.op, '/')


if __name__ == '__main__':
  unittest.main()
