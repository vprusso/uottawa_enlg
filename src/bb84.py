"""BB84 monogamy-of-entanglement game."""
import numpy as np

from toqito.states import basis
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame


def bb84_extended_nonlocal_game():
    """Define the BB84 extended nonlocal game."""
    e_0, e_1 = basis(2, 0), basis(2, 1)
    e_p = (e_0 + e_1) / np.sqrt(2)
    e_m = (e_0 - e_1) / np.sqrt(2)

    dim = 2
    num_alice_out, num_bob_out = 2, 2
    num_alice_in, num_bob_in = 2, 2

    pred_mat = np.zeros([dim, dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in])
    pred_mat[:, :, 0, 0, 0, 0] = e_0 * e_0.conj().T
    pred_mat[:, :, 1, 1, 0, 0] = e_1 * e_1.conj().T
    pred_mat[:, :, 0, 0, 1, 1] = e_p * e_p.conj().T
    pred_mat[:, :, 1, 1, 1, 1] = e_m * e_m.conj().T

    prob_mat = 1 / 2 * np.identity(2)

    return prob_mat, pred_mat


if __name__ == "__main__":
    prob_mat, pred_mat = bb84_extended_nonlocal_game()
    enlg = ExtendedNonlocalGame(prob_mat, pred_mat, reps=2)

    ent = enlg.quantum_value_lower_bound(iters=2, tol=10e-3)
    unent = enlg.unentangled_value()
    ns = enlg.nonsignaling_value()

    print("Entangled value (lower bound): ", ent)
    print("Unentangled value: ", unent)
    print("Non-signaling value: ", ns)

