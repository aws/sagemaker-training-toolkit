class XlaRuntimeError(Exception):
    """dummy XlaRuntimeError class to throw. Unable to import the actual
    XlaRuntimeError class defined in tensorflow/compiler/xla/python/xla_client.py module.
    """


raise XlaRuntimeError("Throwing the dummy exception to simulate the error.")
