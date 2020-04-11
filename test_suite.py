from environment_test import EnvironmentMethods
from DQN_agent_test import DQnAgentMethods
import unittest

def suite():
    suite = unittest.TestSuite()
    suite.addTest(DQnAgentMethods())
    suite.addTest(EnvironmentMethods())
    return suite

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DQnAgentMethods)
    unittest.TextTestRunner().run(suite)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EnvironmentMethods)
    unittest.TextTestRunner().run(suite)