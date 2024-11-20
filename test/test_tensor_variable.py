import unittest
import torch
from torch._dynamo.variables.tensor import TensorVariable
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.misc import GetAttrVariable
from torch._dynamo.variables.builder import wrap_fx_proxy

class TestTensorVariable(unittest.TestCase):
    def setUp(self):
        self.tx = InstructionTranslator.current_tx()

    def test_tensor_returning_method(self):
        tensor = torch.tensor([1, 2, 3])
        tensor_var = TensorVariable(wrap_fx_proxy(self.tx, self.tx.output.create_proxy("placeholder", "tensor", (), {})))
        result = tensor_var.call_method(self.tx, "clone", [], {})
        self.assertIsInstance(result, TensorVariable)

    def test_non_tensor_returning_method(self):
        tensor = torch.tensor([1, 2, 3])
        tensor_var = TensorVariable(wrap_fx_proxy(self.tx, self.tx.output.create_proxy("placeholder", "tensor", (), {})))
        with self.assertRaises(NotImplementedError):
            tensor_var.call_method(self.tx, "item", [], {})

    def test_unsupported_method(self):
        tensor = torch.tensor([1, 2, 3])
        tensor_var = TensorVariable(wrap_fx_proxy(self.tx, self.tx.output.create_proxy("placeholder", "tensor", (), {})))
        with self.assertRaises(NotImplementedError):
            tensor_var.call_method(self.tx, "unsupported_method", [], {})

if __name__ == "__main__":
    unittest.main()
