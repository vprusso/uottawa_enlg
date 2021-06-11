"""CHSH nonlocal game."""
import numpy as np

from toqito.states import basis
from toqito.nonlocal_games import NonlocalGame

# Define the number of inputs/outputs for each player.
num_alice_inputs, num_alice_outputs = 2, 2
num_bob_inputs, num_bob_outputs = 2, 2

# The probability matrix and predicate matrix define any nonlocal game.
prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))

# Populate predicate with "1" when winning condition is achieved and with zero
# otherwise.
for a_alice in range(num_alice_outputs):
    for b_bob in range(num_bob_outputs):
        for x_alice in range(num_alice_inputs):
            for y_bob in range(num_bob_inputs):
                if a_alice ^ b_bob == x_alice * y_bob:
                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

# Instantiate `NonlocalGame` object from `prob_mat` and `pred_mat`.
chsh = NonlocalGame(prob_mat, pred_mat)

# Calculate the quantum and classical value of the CHSH game.
print("Quantum value: ", chsh.quantum_value_lower_bound())
print("Classical value: ", chsh.classical_value())
