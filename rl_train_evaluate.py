import copy
import time
from tqdm import tqdm

import traceback
from pathlib import Path
import joblib

import multiprocessing
from multiprocessing import Pool, cpu_count #, Manager


import numpy as np
import pandas as pd

from game import State
from rl_models import BOARDSIZE, QLearningAgent, SarsaAgent, ExpectedSarsaAgent, DoubleQLearningAgent, DoubleSarsaAgent, DoubleExpectedSarsaAgent

from mcts import MCTSAgent
from dual_network import NNEvaluator, load_model
from benchmark_agents import RandomAgent, GreedyDistanceAgent


#BOARDSIZE = 5
WALLS = 1

SEED = 0
PROCESSES = cpu_count() - 1

DRAW_REWARD = -0.2 # penalty for being risk-averse
WIN_REWARD = 1.0
LOSS_REWARD = -1.0

MODELS_DIR = Path(f"models_{BOARDSIZE}x{BOARDSIZE}_with_{WALLS}_walls")
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
BEST_DIR = MODELS_DIR / "best"

REFERENCE_MODEL_PATH = Path("models_5x5/supervised_extended.pt")
REFERENCE_MODEL_SCORES_PATH = BEST_DIR / "scores"
REFERENCE_MODEL_NUM_SIMULATIONS = 1

GLOBAL_EVAL_AGENT_A = None
GLOBAL_EVAL_AGENT_B = None

def make_run_dir(run_name=None):
    if run_name is None:
        run_name = time.strftime("%Y%m%d_%H%M%S")
    return CHECKPOINT_DIR / run_name

def find_latest_epoch(run_dir):
    if not run_dir.exists():
        return None

    epoch_dirs = sorted(run_dir.glob("epoch_*"))
    if not epoch_dirs:
        return None

    return int(epoch_dirs[-1].name.split("_")[1])

def save_checkpoint(run_dir, worker_id, agent, epoch):
    epoch_dir = run_dir / f"worker_{worker_id}" / f"epoch_{epoch:06d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    agent.save(epoch_dir / "agent.pkl")

def load_checkpoint(run_dir, worker_id, agent):
    worker_dir = run_dir / f"worker_{worker_id}"
    latest_epoch = find_latest_epoch(worker_dir)
    if latest_epoch is None:
        return agent, 0
    print(f"[Worker {worker_id}] Resuming from epoch {latest_epoch}")
    agent.load(worker_dir / f"epoch_{latest_epoch:06d}" / "agent.pkl")
    return agent, latest_epoch

def save_compressed_to_uncompressed(compressed_path, uncompressed_path=None):
    compressed_path = Path(compressed_path)
    if uncompressed_path is None:
        if compressed_path.stem.endswith("_compressed"):
            uncompressed_path = (compressed_path.with_name(compressed_path.stem[:-11] + compressed_path.suffix) if compressed_path.stem.endswith("_compressed") else
                                 compressed_path.with_name(compressed_path.stem + "_uncompressed" + compressed_path.suffix))

    print('Loading compressed agent')
    data = joblib.load(compressed_path)
    print('Saving uncompressed agent')
    joblib.dump(data, uncompressed_path)
    return Path(uncompressed_path)

def save_uncompressed_to_compressed(uncompressed_path, compressed_path=None):
    uncompressed_path = Path(uncompressed_path)
    if compressed_path is None:
        compressed_path = uncompressed_path.with_name(uncompressed_path.stem + "_compressed" + uncompressed_path.suffix)

    print('Loading uncompressed agent')
    data = joblib.load(uncompressed_path)
    print('Saving compressed agent')
    joblib.dump(data, compressed_path, compress=('gzip', 3))
    return Path(compressed_path)



# def init_worker(agent1, agent2):
#     global GLOBAL_AGENT1
#     global GLOBAL_AGENT2
#
#     GLOBAL_AGENT1 = copy.deepcopy(agent1)
#     GLOBAL_AGENT2 = copy.deepcopy(agent2)


def self_play_episode_alphazero(agent, frozen_opp, boardsize, walls):
    """AlphaZero-style: only `agent` learns; `frozen_opp` plays greedily without updating."""
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    learner_player = 1 if agent.rng.random() < 0.5 else 2

    while not state.is_finished():
        current_player = state.get_current_player()
        is_learner = current_player == learner_player

        action = agent.select_action(state, training=True) if is_learner else frozen_opp.select_action(state, training=False)
        next_state = state.next(action)

        if is_learner:
            is_terminal = next_state.is_finished()
            if is_terminal:
                winner = next_state.winner()
                reward = DRAW_REWARD if winner == 0 else (WIN_REWARD if winner == current_player else LOSS_REWARD)
            else:
                reward = 0.0
            agent.update(state, action, reward, next_state, is_terminal)

        state = next_state

def play(agent_a, agent_b, boardsize, walls):
    state = State(boardsize=boardsize, walls_p1=walls, walls_p2=walls)
    while not state.is_finished():
        agent = agent_a if state.get_current_player() == 1 else agent_b
        state = state.next(agent.select_action(state, training=False))

    return state.winner() #"depth": state.depth} # depth is used to determine how long the game has progressed and which player's turn it is

def init_eval_worker(agent_a, agent_b):
    global GLOBAL_EVAL_AGENT_A
    global GLOBAL_EVAL_AGENT_B

    GLOBAL_EVAL_AGENT_A = agent_a
    GLOBAL_EVAL_AGENT_B = agent_b

def evaluate_one(args):
    i, boardsize, walls = args

    global GLOBAL_EVAL_AGENT_A
    global GLOBAL_EVAL_AGENT_B

    try:
        if i % 2 == 0:
            winner = play(GLOBAL_EVAL_AGENT_A, GLOBAL_EVAL_AGENT_B, boardsize, walls)
            if winner == 1:
                return "A"
            elif winner == 2:
                return "B"
            return "draw"

        winner = play(GLOBAL_EVAL_AGENT_B, GLOBAL_EVAL_AGENT_A, boardsize, walls)
        if winner == 1:
            return "B"
        elif winner == 2:
            return "A"
        return "draw"
    
    except Exception as _:
        print(f"Error: worker={i}\n{traceback.format_exc()}")

def evaluate(agent_a, agent_b, boardsize, walls, games, eval_mode=False, processes=PROCESSES):
    wins = {"A": 0, "B": 0, "draw": 0}

    if not eval_mode:
        for i in range(games):
            if i % 2 == 0:
                winner = play(agent_a, agent_b, boardsize, walls)
                if winner == 1:
                    wins["A"] += 1
                elif winner == 2:
                    wins["B"] += 1
                else:
                    wins["draw"] += 1
            else:
                winner = play(agent_b, agent_a, boardsize, walls)
                if winner == 1:
                    wins["B"] += 1
                elif winner == 2:
                    wins["A"] += 1
                else:
                    wins["draw"] += 1

        return wins

    multiprocessing.freeze_support()
    with Pool(processes=processes, initializer=init_eval_worker, initargs=(agent_a, agent_b)) as pool:
        results = list(tqdm(pool.imap_unordered(evaluate_one, [(i, boardsize, walls) for i in range(games)]), total=games, dynamic_ncols=True))

    filtered = [r for r in results if r is not None]
    print(f"Completed: {len(filtered)}/{len(results)} tasks in evaluation")

    for result in filtered:
        wins[result] += 1

    return wins

def train_worker(args):
    worker_id, run_dir, agent, epochs, boardsize, walls, checkpoint_every, frozen_refresh_period = args

    rng = np.random.default_rng(SEED + worker_id)
    agent.set_seed(int(rng.integers(0, 2**31 - 1)))

    try:
        agent, start_epoch = load_checkpoint(run_dir, worker_id, agent)
        frozen_opp = copy.deepcopy(agent)

        iterator = tqdm(range(start_epoch, epochs), leave=False) if worker_id == 0 else range(start_epoch, epochs)
        for epoch in iterator:
            self_play_episode_alphazero(agent, frozen_opp, boardsize, walls)

            if (epoch + 1) % frozen_refresh_period == 0:
                frozen_opp = copy.deepcopy(agent)

            if (epoch + 1) % checkpoint_every == 0:
                save_checkpoint(run_dir, worker_id, agent, epoch + 1)
                print(f"\nCheckpoint saved at epoch {epoch + 1}/{epochs}")

        return {"worker_id": worker_id, "agent": agent}

    except Exception as _:
        print(f"Error: worker={worker_id}, agent={agent}\n{traceback.format_exc()}")


def _outcome_str(winner, agent_player):
    """Human-readable outcome from the perspective of agent_player."""
    if winner == 0:
        return "draw"
    return "win" if winner == agent_player else "loss"

def update_global_best(agent, boardsize, walls,
                       benchmark_sigma=True, benchmark_random=True, benchmark_greedy=True,
                       games_sigma=None, games_random=None, games_greedy=None,
                       games_random_default=400, games_greedy_default=10):
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    agent_name = agent.__class__.__name__

    agent_path            = BEST_DIR / f"{agent_name}.pkl"
    agent_compressed_path = BEST_DIR / f"{agent_name}_compressed.pkl"

    # Informational: 2 games vs previously saved best (1 as P1, 1 as P2)
    existing_path = agent_path if agent_path.exists() else (agent_compressed_path if agent_compressed_path.exists() else None)
    if existing_path is not None:
        old_agent = copy.deepcopy(agent)
        old_agent.load(existing_path)
        w_as_p1 = play(agent, old_agent, boardsize, walls)
        w_as_p2 = play(old_agent, agent, boardsize, walls)
        print(f"[{agent_name}] vs old self — as P1: {_outcome_str(w_as_p1, 1)}  |  as P2: {_outcome_str(w_as_p2, 2)}")

    # Benchmarks
    score = None
    score_vs_random = None
    score_vs_greedy = None

    if benchmark_sigma:
        model = load_model(REFERENCE_MODEL_PATH)
        model.eval()
        reference_agent = MCTSAgent(evaluator=NNEvaluator(model), num_simulations=REFERENCE_MODEL_NUM_SIMULATIONS, training=False)
        n = games_sigma if games_sigma is not None else games_random_default
        result = evaluate(agent, reference_agent, boardsize, walls, n, eval_mode=True)
        score = (result["A"] + 0.5 * result["draw"]) / (result["A"] + result["B"] + result["draw"])

    if benchmark_random:
        n = games_random if games_random is not None else games_random_default
        n_half = max(1, n // 2)
        rng_agent = RandomAgent()
        p1_wins = sum((lambda w: 1.0 if w == 1 else 0.5 if w == 0 else 0.0)(play(agent, rng_agent, boardsize, walls)) for _ in range(n_half))
        p2_wins = sum((lambda w: 1.0 if w == 2 else 0.5 if w == 0 else 0.0)(play(rng_agent, agent, boardsize, walls)) for _ in range(n_half))
        score_vs_random_p1 = p1_wins / n_half
        score_vs_random_p2 = p2_wins / n_half
        score_vs_random = (score_vs_random_p1 + score_vs_random_p2) / 2.0
        print(f"[Random]      {agent_name} vs random — as P1: {score_vs_random_p1:.3f}  |  as P2: {score_vs_random_p2:.3f}  (n={n_half} each)")

    if benchmark_greedy:
        greedy = GreedyDistanceAgent()
        w_as_p1 = play(agent, greedy, boardsize, walls)
        w_as_p2 = play(greedy, agent, boardsize, walls)
        score_vs_greedy = (int(w_as_p1 == 1) + int(w_as_p2 == 2)) / 2.0
        print(f"[Greedy dist] {agent_name} vs greedy — as P1: {_outcome_str(w_as_p1, 1)}  |  as P2: {_outcome_str(w_as_p2, 2)}")

    if score is not None:
        print(f"[Sigma ref]   {agent_name} vs reference agent:     {score:.4f}")

    # Always save — trust the training process
    agent.save(agent_path)

    if Path(REFERENCE_MODEL_SCORES_PATH).exists():
        df = pd.read_csv(REFERENCE_MODEL_SCORES_PATH)
    else:
        df = pd.DataFrame(columns=["model", "score_vs_sigma", "score_vs_random", "score_vs_greedy"])
    row = df[df["model"] == agent_name]

    new_row = {"model": agent_name}
    if score is not None:           new_row["score_vs_sigma"]  = score
    if score_vs_random is not None: new_row["score_vs_random"] = score_vs_random
    if score_vs_greedy is not None: new_row["score_vs_greedy"] = score_vs_greedy

    if row.empty:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        for col, val in new_row.items():
            if col != "model":
                df.loc[df["model"] == agent_name, col] = val
    df.to_csv(REFERENCE_MODEL_SCORES_PATH, index=False)

    print(f"[{agent_name}] Saved: {agent_path}")


def train_population(agent_classes, epochs=5000, boardsize=BOARDSIZE, walls=WALLS,
                     processes=PROCESSES, checkpoint_every=20000, frozen_refresh_period=500,
                     benchmark_sigma=True, benchmark_random=True, benchmark_greedy=True,
                     games_sigma=None, games_random=None, games_greedy=None):
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    # Load best saved model for each agent class (fresh if none exists)
    agents = []
    for cls in agent_classes:
        agent = cls(boardsize=boardsize)
        path            = BEST_DIR / f"{cls.__name__}.pkl"
        compressed_path = BEST_DIR / f"{cls.__name__}_compressed.pkl"
        if path.exists():
            agent.load(path)
            print(f"Loaded {cls.__name__}")
        elif compressed_path.exists():
            agent.load(compressed_path)
            print(f"Loaded {cls.__name__} (compressed)")
        else:
            print(f"No saved model for {cls.__name__}, starting fresh.")
        agents.append(agent)

    run_dir = make_run_dir()
    print(f"Run dir: {run_dir}")

    worker_args = [
        (i, run_dir, copy.deepcopy(agents[i]), epochs, boardsize, walls, checkpoint_every, frozen_refresh_period)
        for i in range(len(agents))
    ]

    multiprocessing.freeze_support()
    with Pool(processes=min(processes, len(agents))) as pool:
        results_raw = list(tqdm(pool.imap_unordered(train_worker, worker_args), total=len(worker_args), dynamic_ncols=True))

    results = [r for r in results_raw if r is not None]
    print(f"Completed: {len(results)}/{len(results_raw)} training runs")

    if not results:
        print("All workers failed — skipping tournament and model update.")
        return

    # Inter-type tournament: 2 deterministic games per pair (1 per side)
    if len(results) > 1:
        print("\nInter-type tournament (2 games per pair):")
        t_scores = {r["worker_id"]: 0.0 for r in results}
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                ai, aj = results[i]["agent"], results[j]["agent"]
                wi, wj = results[i]["worker_id"], results[j]["worker_id"]
                w1 = play(ai, aj, boardsize, walls)
                w2 = play(aj, ai, boardsize, walls)
                si = (1.0 if w1 == 1 else 0.5 if w1 == 0 else 0.0) + (1.0 if w2 == 2 else 0.5 if w2 == 0 else 0.0)
                t_scores[wi] += si
                t_scores[wj] += 2.0 - si
                print(f"  {ai.__class__.__name__} vs {aj.__class__.__name__}: {si:.1f}-{2.0-si:.1f}")

        print("\nStandings:")
        for r in sorted(results, key=lambda r: t_scores[r["worker_id"]], reverse=True):
            print(f"  {r['agent'].__class__.__name__}: {t_scores[r['worker_id']]:.1f} pts")

    # Benchmark and independently save each agent
    for result in results:
        update_global_best(result["agent"], boardsize, walls,
                           benchmark_sigma=benchmark_sigma, benchmark_random=benchmark_random,
                           benchmark_greedy=benchmark_greedy,
                           games_sigma=games_sigma, games_random=games_random, games_greedy=games_greedy)


if __name__ == "__main__":
    NUM_CYCLES = 20
    EPOCHS_PER_CYCLE = 20000
    AGENT_CLASSES = [
        QLearningAgent, SarsaAgent, ExpectedSarsaAgent,
        DoubleQLearningAgent, DoubleSarsaAgent, DoubleExpectedSarsaAgent,
    ]

    for cycle in range(1, NUM_CYCLES + 1):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle}/{NUM_CYCLES}")
        print(f"{'='*60}")
        train_population(AGENT_CLASSES, epochs=EPOCHS_PER_CYCLE, benchmark_sigma=False,
                         games_random=400, games_greedy=2)