import runpy
import os
import mbtr

mbtr.splash()

print('Commencing tests...')
k = os.path.dirname(os.path.realpath(__file__))
os.chdir(k)
print('standalone tests...')
runpy.run_path('tests/standalone_tests.py', run_name='__main__')
